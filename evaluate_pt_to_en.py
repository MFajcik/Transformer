import gc
from torchtext.data import Dataset
from tqdm import tqdm

from playground import *
from os.path import join
from socket import gethostname

from train_pt_to_en import PT_TO_EN_dataset, get_BLEU
from util import get_timestamp, gpu_mem_restore, SEP_TOKEN
from nltk.translate.bleu_score import sentence_bleu as nltk_sentence_bleu, SmoothingFunction
from sacrebleu import corpus_bleu as sacrebleu_corpus_bleu, BLEU
from mosestokenizer import MosesDetokenizer

BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"
PATH_TO_DATA = ".data/pt-to-en/data"
REF_FILE = os.path.join(PATH_TO_DATA, "val/text.en")
BLEU_LOWERCASE = True
BEAM_SIZE = 5
DO_NOT_EVAL_SAVE_EPOCHS = 50


def eval_PT_to_EN(p_to_model, method):
    """
    Train on  Portuguese-English Translation task
    """

    # spacy_en = spacy.load('en')
    # spacy_pt = spacy.load('pt')

    # I have pre-tokenized the outputs, all tokens are split via SEP_TOKEN
    def tokenize_pt(text):
        return [t for t in str.split(text, sep=SEP_TOKEN) if t != ""]
        # return [tok.text for tok in spacy_pt.tokenizer(text)]

    def tokenize_en(text):
        return [t for t in str.split(text, sep=SEP_TOKEN) if t != ""]
        # return [tok.text for tok in spacy_en.tokenizer(text)]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    SRC = data.Field(tokenize=tokenize_pt, pad_token=BLANK_WORD)
    TGT = data.Field(tokenize=tokenize_en, init_token=BOS_WORD,
                     eos_token=EOS_WORD, pad_token=BLANK_WORD)
    MAX_LEN = 512
    MAX_LEN_BLEU = 100
    train, val, test = PT_TO_EN_dataset.splits(path=PATH_TO_DATA, fields=(SRC, TGT),
                                               # use only examples shorter than 512
                                               filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
                                                                     len(vars(x)['trg']) <= MAX_LEN)

    MIN_VOCAB_FREQ = 2
    SRC.build_vocab(train.src, min_freq=MIN_VOCAB_FREQ)
    TGT.build_vocab(train.trg, min_freq=MIN_VOCAB_FREQ)

    logging.info(f"Vocab size SRC: {len(SRC.vocab)}")
    logging.info(f"Vocab size TGT: {len(TGT.vocab)}")

    pad_idx = TGT.vocab.stoi["<blank>"]

    criterion = LabelSmoothing(vocab_size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    VAL_BS = 1024 + 256
    BLEU_BS = 512
    # These examples are not shuffled
    valid_iter = DataIterator(val, batch_size=VAL_BS, device=device,
                              repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                              batch_size_fn=batch_size_fn, train=False)
    test_iter = DataIterator(test, batch_size=BLEU_BS, device=device,
                             repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                             batch_size_fn=batch_size_fn, train=False)

    model = torch.load(open(p_to_model, "rb"), map_location=device)
    model.eval()
    val_loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter),
                         model,
                         SingleGPULossCompute(model.generator, criterion, opt=None))
    test_loss = run_epoch((rebatch(pad_idx, b) for b in test_iter),
                          model,
                          SingleGPULossCompute(model.generator, criterion, opt=None))

    val_bleu = -1
    val_bleu = get_BLEU((rebatch(pad_idx, b) for b in valid_iter), model, SRC=SRC,
                        TGT=TGT,
                        max_len=MAX_LEN_BLEU,
                        method=method,
                        start_symbol=TGT.vocab.stoi["<s>"], total_batches=len([x for x in valid_iter]))
    test_bleu = get_BLEU((rebatch(pad_idx, b) for b in test_iter), model, SRC=SRC,
                         TGT=TGT,
                         max_len=MAX_LEN_BLEU,
                         method=method,
                         start_symbol=TGT.vocab.stoi["<s>"], total_batches=len([x for x in test_iter]))

    return val_loss, val_bleu, test_loss, test_bleu


# score=34.96939944413386,
# counts=[29569, 18255, 12013, 8006],
# totals=[43110, 41088, 39066, 37071],
# precisions=[68.58965437253538, 44.42903037383178, 30.750524752982134, 21.596396104771923],
# bp=0.9271464880101749,
# sys_len=43110,
# ref_len=46371
import re

if __name__ == "__main__":
    ofs = len("pt_to_en_E")
    savedpoints = sorted(os.listdir("saved"), key=lambda x: int(re.sub("[^0-9]", "", x[ofs:ofs + 3])), reverse=True)

    setup_logging(os.path.basename(sys.argv[0]).split(".")[0],
                  logpath="logs/",
                  config_path="configurations/logging.yml")
    method = "sacreBLEU"
    for savedpoint in savedpoints:
        logging.info("Evaluated:")
        logging.info(savedpoint)

        val_loss, val_bleu, test_loss, test_bleu = eval_PT_to_EN(f"saved/{savedpoint}", method)
        logging.info(f"Validation Loss: {val_loss}")

        val_bleu_score = val_bleu if not hasattr(val_bleu, 'score') else val_bleu.score
        test_bleu_score = test_bleu if not hasattr(test_bleu, 'score') else test_bleu.score

        logging.info("VAL LOSS:")
        logging.info(val_loss)
        logging.info("-------------")
        logging.info(f"VAL BLEU: {val_bleu_score}")
        logging.info("-------------")

        logging.info("TEST_LOSS:")
        logging.info(test_loss)
        logging.info("-------------")
        logging.info(f"TEST BLEU: {test_bleu_score}")
        logging.info("-------------")
