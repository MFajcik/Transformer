from torchtext.data import Dataset

from playground import *
from os.path import join


class PT_TO_EN_dataset(Dataset):
    """The PT-to-END translation task"""

    def __init__(self, path, fields, srcfile="text.pt", trgfile="text.en", **kwargs):
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]
        src_path = join(path, srcfile)
        trg_path = join(path, trgfile)
        examples = []
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
        """Create dataset objects for splits of the IWSLT dataset.

        Arguments:
            fields: A tuple containing the fields that will be used for data
                in each language.
        """

        train_path = join(path, train)
        validation_path = join(path, validation)
        test_path = join(path, test)

        train_data = cls(train_path, fields, **kwargs)
        val_data = cls(validation_path, fields, **kwargs)
        test_data = cls(test_path, fields, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data))


def train_PT_to_EN():
    """
    Train on  Portuguese-English Translation task
    """
    PATH_TO_DATA = "../../Work/NLP/Corpus/pt_to_en_translation/data"
    spacy_pt = spacy.load('pt')
    spacy_en = spacy.load('en')

    def tokenize_pt(text):
        return [tok.text for tok in spacy_pt.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"
    SRC = data.Field(tokenize=tokenize_pt, pad_token=BLANK_WORD)
    TGT = data.Field(tokenize=tokenize_en, init_token=BOS_WORD,
                     eos_token=EOS_WORD, pad_token=BLANK_WORD)
    MAX_LEN = 100
    train, val, test = PT_TO_EN_dataset.splits(path=PATH_TO_DATA, fields=(SRC, TGT),
                                               filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
                                                                     len(vars(x)['trg']) <= MAX_LEN)
    MIN_FREQ = 2
    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train.trg, min_freq=MIN_FREQ)

    pad_idx = TGT.vocab.stoi["<blank>"]
    model = create_transformer_model(len(SRC.vocab), len(TGT.vocab), N=6).to(device)
    criterion = LabelSmoothing(vocab_size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    BATCH_SIZE = 1024
    # These examples are shuffled
    train_iter = DataIterator(train, batch_size=BATCH_SIZE, device=device,
                              repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                              batch_size_fn=batch_size_fn, train=True)
    # These examples are not shuffled
    valid_iter = DataIterator(val, batch_size=BATCH_SIZE, device=device,
                              repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                              batch_size_fn=batch_size_fn, train=False)

    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    for epoch in range(10):
        model.train()
        run_epoch((rebatch(pad_idx, b) for b in train_iter),
                  model,
                  SingleGPULossCompute(model.generator, criterion, model_opt))
        model.eval()
        loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter),
                         model,
                         SingleGPULossCompute(model.generator, criterion, opt=None))
        print(loss)
        logging.info(f"Validation Loss: {loss}")


if __name__ == "__main__":
    setup_logging(os.path.basename(sys.argv[0]).split(".")[0],
                  logpath="logs/",
                  config_path="configurations/logging.yml")
    train_PT_to_EN()
