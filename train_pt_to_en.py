import gc
from torchtext.data import Dataset
from tqdm import tqdm

from playground import *
from os.path import join
from socket import gethostname
from util import get_timestamp, gpu_mem_restore, SEP_TOKEN
from nltk.translate.bleu_score import sentence_bleu as nltk_sentence_bleu, SmoothingFunction
#from sacrebleu import corpus_bleu as sacrebleu_corpus_bleu, BLEU
#from mosestokenizer import MosesDetokenizer

BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"
PATH_TO_DATA = ".data/pt-to-en/data"
REF_FILE = os.path.join(PATH_TO_DATA, "val/text.en")
BLEU_LOWERCASE = True
BEAM_SIZE = 5
DO_NOT_EVAL_SAVE_EPOCHS = 0


### Get dataset here:
# https://github.com/srvk/how2-dataset
###

class PT_TO_EN_dataset(Dataset):
    """The PT-to-END translation task"""

    def __init__(self, path, fields, srcfile="text.pt.tokenized", trgfile="text.en.tokenized", verbose=True, **kwargs):
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]
        src_path = join(path, srcfile)
        trg_path = join(path, trgfile)
        examples = []
        if verbose:
            logging.info(f"Loading files \n{src_path}\n{trg_path}")
        with open(src_path, mode='r', encoding='utf-8') as src_file, \
                open(trg_path, mode='r', encoding='utf-8') as trg_file:
            for src_line, trg_line in zip(src_file, trg_file):
                src_line, trg_line = src_line.strip(), trg_line.strip()
                if src_line != '' and trg_line != '':
                    examples.append(data.Example.fromlist(
                        [src_line, trg_line], fields))

        super(PT_TO_EN_dataset, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, path, fields, train='train', validation='val', test='dev5', **kwargs):
        """Create dataset objects for splits of the How2Videos dataset.

        Arguments:
            fields: A tuple containing the fields that will be used for data
                in each language.
        """

        train_path = join(path, train)
        validation_path = join(path, validation)
        test_path = join(path, test)

        logging.info("Loading data:")
        logging.info(f"Training set: {train_path}")
        logging.info(f"Validation set: {validation_path}")
        logging.info(f"Test set: {test_path}")
        train_data = cls(train_path, fields, **kwargs)
        val_data = cls(validation_path, fields, **kwargs)
        test_data = cls(test_path, fields, **kwargs)
        return train_data,val_data,test_data


def get_BLEU_nltk(data_iter, model, SRC, TGT, total_batches, decoding="greedy", fname="translation", **kwargs):

    assert model.training ==  False
    def to_tokenized_text(tensor):
        strs = []
        for example_idx in range(len(tensor)):
            s = []
            for i in tensor[example_idx]:
                token = TGT.vocab.itos[i]
                if token == BOS_WORD:
                    continue
                if token == EOS_WORD:
                    break
                s.append(token)
                assert token != BLANK_WORD
            strs.append(s)
        return strs

    bleu_acc = 0
    N = 0
    # pbar = tqdm(total=total_batches)
    for i, batch in enumerate(data_iter):
        # Debug
        # print("-" * 10 + "SRC" + "-" * 10)
        # print("\n".join(totext(batch.src, src_vocab)))
        # print("-" * 10 + "TGT" + "-" * 10)
        # print("\n".join(totext(batch.trg, tgt_vocab)))
        # print("-" * 30)

        if decoding == "greedy":
            # Greedy decoding
            decoded = greedy_decode(model, batch.src, batch.src_mask, **kwargs)
            candidate_tokens = to_tokenized_text(decoded)
        elif decoding == "beam_search":
            # Beam Search
            decoded_hypotheses, logprobs = beam_search(model, batch.src, batch.src_mask, kwargs["max_len"],
                                                       TGT.vocab.stoi["<blank>"], TGT.vocab.stoi[BOS_WORD],
                                                       TGT.vocab.stoi[EOS_WORD], BEAM_SIZE, torch.device("cuda:0"))
            most_probable_hypotheses = [h[0] for h in decoded_hypotheses]
            candidate_tokens = to_tokenized_text(most_probable_hypotheses)
        # with Beam BLEU: 0.6100825122348087

        # Debug
        # decoded_text = totext(decoded, tgt_vocab)
        # get rid of special tokens
        # tmp = []
        # for s in decoded_text:
        #    idx = s.find("</s>")
        #    tmp.append(s[:idx] if idx > 0 else s)
        # decoded_text = tmp
        # print("-" * 10 + "DEC" + "-" * 10)
        # print("\n".join(decoded_text))
        # print("-" * 30)

        ref_tokens = to_tokenized_text(batch.trg)
        assert len(candidate_tokens) == len(ref_tokens)

        # Debug
        # print("-" * 10 + "GT" + "-" * 10)
        # for ref in ref_tokens: print(ref)
        # print("-" * 30)

        # print("-" * 10 + "PREDICTION" + "-" * 10)
        # for c in candidate_tokens: print(c)
        # print("-" * 30)
        chencherry = SmoothingFunction().method4

        for k in range(len(candidate_tokens)):
            # usually BLEU 1 to 4 is averaged, but this is problem if the sequence is too short
            # if len(candidate_tokens[k]) < 4:
            #   weights = (1 / len(candidate_tokens[k]) for _ in range(len(candidate_tokens[k])))\

            # auto_reweigh parameter should actually do t he same as if calling above 2 lines!
            # print("REF:")
            # print(ref_tokens[k])
            # print("CAND:")
            # print(candidate_tokens[k])
            try:
                sent_bleu = nltk_sentence_bleu([ref_tokens[k]], candidate_tokens[k], auto_reweigh=True,
                                               smoothing_function=chencherry)
            except ZeroDivisionError as ze: # if 0 length sentence has been decoded...
                sent_bleu = 0
            bleu_acc += sent_bleu
            N += 1
        # pbar.set_description(f"BLEU: {bleu_acc / N:.2f}")
        # pbar.update(1)
    if N == 0:
        return 0
    return bleu_acc / N


def get_BLEU_sacreBLEU(data_iter, model, SRC, TGT, total_batches, rfname="ground_truth",
                       tfname="translation",
                       **kwargs):
    assert model.training ==  False
    # pbar = tqdm(total=total_batches)
    # hypotheses = "outputs/translation_pcfajcik_2019-04-12_10:39"
    # references = "outputs/ground_truth_pcfajcik_2019-04-12_10:39"
    references = f"outputs/{rfname}_{gethostname()}_{get_timestamp()}"
    hypotheses = f"outputs/{tfname}_{gethostname()}_{get_timestamp()}"
    detokenize = MosesDetokenizer('en')
    with open(hypotheses, mode="w") as hf:
        with open(references, mode="w") as rf:
            for i, batch in enumerate(data_iter):
                # Debug
                # print("-" * 10 + "SRC" + "-" * 10)
                # print("\n".join(totext(batch.src, src_vocab)))
                # print("-" * 10 + "TGT" + "-" * 10)
                # print("\n".join(totext(batch.trg, tgt_vocab)))
                # print("-" * 30)
                ground_truth = totext(batch.trg, TGT.vocab, sep=SEP_TOKEN)

                # Greedy decode
                # BLEU(score=59.28653001528331, counts=[34404, 26328, 20732, 16451],
                #                  totals=[42856, 40834, 38812, 36816],
                #                  precisions=[80.27814075042001, 64.47568202968115, 53.416469133257756, 44.68437635810517],
                #                  bp=1.0, sys_len=42856, ref_len=42327)
                # decoded = greedy_decode(model, batch.src, batch.src_mask, **kwargs)
                # decoded_text = totext(decoded, TGT.vocab)

                # Beam search
                # h5 = BLEU(score=60.81557819840607, counts=[34450, 26529, 21028, 16806],
                #           totals=[41803, 39781, 37759, 35763],
                #           precisions=[82.41035332392413, 66.68761469043011, 55.69003416404036, 46.99270195453401],
                #           bp=0.9875432501681118, sys_len=41803, ref_len=42327)
                # with beam 30 BLEU: BLEU(score=60.70840003601362

                # With correct detokenization
                # BLEU(score=59.81264905560044
                decoded_hypotheses, logprobs = beam_search(model, batch.src, batch.src_mask, kwargs["max_len"],
                                                           TGT.vocab.stoi["<blank>"], TGT.vocab.stoi[BOS_WORD],
                                                           TGT.vocab.stoi[EOS_WORD], BEAM_SIZE, torch.device("cuda:0"))

                most_probable_hypotheses = [h[0] for h in decoded_hypotheses]
                decoded_text = totext(most_probable_hypotheses, TGT.vocab, sep=SEP_TOKEN)

                # get rid of special tokens
                clean_decoded_text = []
                for s in decoded_text:
                    eos_idx = s.find(EOS_WORD)
                    cleaned = s[:eos_idx] if eos_idx > 0 else s
                    clean_decoded_text.append(detokenize(cleaned.strip().split(sep=SEP_TOKEN)))

                clean_ground_truth_text = []
                for s in ground_truth:
                    eos_idx = s.find(EOS_WORD)
                    bos_idx = s.find(BOS_WORD)
                    if bos_idx > -1 and eos_idx > -1:
                        clean_s = s[bos_idx + len(BOS_WORD):eos_idx]
                    elif eos_idx > -1:
                        clean_s = s[:eos_idx]
                    elif bos_idx > -1:
                        clean_s = s[bos_idx + len(BOS_WORD):]
                    else:
                        clean_s = s
                    clean_ground_truth_text.append(detokenize(clean_s.strip().split(sep=SEP_TOKEN)))

                if not len(clean_decoded_text) == batch.src.shape[0]:
                    print(len(clean_decoded_text))
                    print(batch.src.shape[0])

                    print(clean_decoded_text)
                    print(batch.src)

                assert len(clean_decoded_text) == len(clean_ground_truth_text) == batch.src.shape[0]
                hf.write("\n".join(clean_decoded_text) + "\n")
                rf.write("\n".join(clean_ground_truth_text) + "\n")
                # pbar.update(1)
    with open(hypotheses) as hypf:
        with open(references) as reff:
            return sacrebleu_corpus_bleu(hypf.read().split("\n")[:-1], [reff.read().split("\n")[:-1]],
                                         lowercase=BLEU_LOWERCASE)


@gpu_mem_restore
def get_BLEU(*args, method="sacreBLEU", **kwargs):
    try:
        if method == "nltk":
            return get_BLEU_nltk(*args, **kwargs)
        elif method == "sacreBLEU":
            return get_BLEU_sacreBLEU(*args, **kwargs)
        else:
            raise NotImplementedError(f"Unknown BLEU evaluation method {method}")
    except Exception as e:
        logging.error(e)


def train_PT_to_EN():
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
    # spacy_de = spacy.load('de')
    # spacy_en = spacy.load('en')
    #
    # def tokenize_pt(text):
    #     return [tok.text for tok in spacy_de.tokenizer(text)]
    #
    # def tokenize_en(text):
    #     return [tok.text for tok in spacy_en.tokenizer(text)]


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
    # train, val, test = datasets.IWSLT.splits(
    #     exts=('.de', '.en'), fields=(SRC, TGT),
    #     filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
    #                           len(vars(x)['trg']) <= MAX_LEN)

    logging.info(f"Training samples: {len(train.examples)}")
    logging.info(f"Validation samples: {len(val.examples)}")
    logging.info(f"Testing samples: {len(test.examples)}")
    MIN_VOCAB_FREQ = 2
    SRC.build_vocab(train.src, min_freq=MIN_VOCAB_FREQ)
    TGT.build_vocab(train.trg, min_freq=MIN_VOCAB_FREQ)

    pad_idx = TGT.vocab.stoi["<blank>"]

    criterion = LabelSmoothing(vocab_size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    TRAIN_BS = 1024 + 256 + 128
    VAL_BS = 1024 + 256
    BLEU_BS = 256
    # These examples are shuffled
    train_iter = DataIterator(train, batch_size=TRAIN_BS, device=device,
                              repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                              batch_size_fn=batch_size_fn, train=True)
    # These examples are not shuffled
    valid_iter = DataIterator(val, batch_size=VAL_BS, device=device,
                              repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                              batch_size_fn=batch_size_fn, train=False)
    test_iter = DataIterator(test, batch_size=BLEU_BS, device=device,
                             repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                             batch_size_fn=batch_size_fn, train=False)
    BLEU_iter_val = DataIterator(val, batch_size=BLEU_BS, device=device,
                                 repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                                 batch_size_fn=batch_size_fn, train=False)
    BLEU_iter_test = DataIterator(test, batch_size=BLEU_BS, device=device,
                                  repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                                  batch_size_fn=batch_size_fn, train=False)
    model = create_transformer_model(len(SRC.vocab), len(TGT.vocab), N=6).to(device)
    # model = torch.load(open("saved/"
    #                         "pt_to_en_E163_BLEU_0.740259065427156_<class 'playground.EncoderDecoder'>L_0.42052939253054017_2019-04-17_10:17_pcknot3.pt",
    #                         "rb"),
    #                  map_location=device)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    epoch = 0
    while True:
        logging.info(f"Starting epoch {epoch}")
        model.train()
        train_loss = 0
        train_loss = run_epoch((rebatch(pad_idx, b) for b in train_iter),
                               model,
                               SingleGPULossCompute(model.generator, criterion, model_opt))
        model.eval()
        with torch.no_grad():
            val_loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter),
                                 model,
                                 SingleGPULossCompute(model.generator, criterion, opt=None))
            test_loss = run_epoch((rebatch(pad_idx, b) for b in test_iter),
                                  model,
                                  SingleGPULossCompute(model.generator, criterion, opt=None))

            bleu_val = bleu_test = -1
            if epoch > DO_NOT_EVAL_SAVE_EPOCHS or epoch == 0:
                bleu_val = get_BLEU((rebatch(pad_idx, b) for b in BLEU_iter_val), model, SRC=SRC,
                                    TGT=TGT,
                                    max_len=MAX_LEN_BLEU,
                                    start_symbol=TGT.vocab.stoi["<s>"], total_batches=len([x for x in BLEU_iter_val]))

                bleu_test = get_BLEU((rebatch(pad_idx, b) for b in BLEU_iter_test), model, SRC=SRC,
                                     TGT=TGT,
                                     max_len=MAX_LEN_BLEU,
                                     start_symbol=TGT.vocab.stoi["<s>"], total_batches=len([x for x in BLEU_iter_test]))
            logging.info(f"Train Loss: {train_loss}")
            logging.info(f"Validation Loss: {val_loss}")
            logging.info(f"Test Loss: {test_loss}")

            process_bleu = lambda bleu: bleu if not hasattr(bleu, 'score') else bleu.score
            bleu_val = process_bleu(bleu_val)
            bleu_test = process_bleu(bleu_test)

            if epoch > DO_NOT_EVAL_SAVE_EPOCHS or epoch == 0:
                model.to(torch.device("cpu"))
                torch.save(model,
                           f"saved/pt_to_en_E{epoch}_BLEU_{bleu_test}_{str(model.__class__)}"
                           f"L_{val_loss}_{get_timestamp()}_{gethostname()}.pt")
                model.to(device)

            logging.info("-------------")
            logging.info(f"VAL BLEU: {bleu_val}")
            logging.info(f"TEST BLEU: {bleu_test}")
            logging.info("-------------")
            epoch += 1


# score=34.96939944413386,
# counts=[29569, 18255, 12013, 8006],
# totals=[43110, 41088, 39066, 37071],
# precisions=[68.58965437253538, 44.42903037383178, 30.750524752982134, 21.596396104771923],
# bp=0.9271464880101749,
# sys_len=43110,
# ref_len=46371
if __name__ == "__main__":
    setup_logging(os.path.basename(sys.argv[0]).split(".")[0],
                  logpath="logs/",
                  config_path="configurations/logging.yml")
    train_PT_to_EN()
