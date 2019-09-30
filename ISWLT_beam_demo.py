import pickle
import wget

from playground import *


#####################################
# About IWSLT dataset:

# These are the data sets for the MT tasks of the evaluation campaigns of IWSLT.
# They are parallel data sets used for building and testing MT systems. They are publicly available
# through the WIT3 website wit3.fbk.eu, see release: 2016-01.
# IWSLT 2016: from/to English to/from Arabic, Czech, French, German
# Data are crawled from the TED website and carry the respective licensing conditions (for training, tuning and testing MT systems).

# Approximately, for each language pair, training sets include 2,000 talks, 200K sentences and 4M tokens per side,
# while each dev and test sets 10-15 talks, 1.0K-1.5K sentences and 20K-30K tokens per side. In each edition,
# the training sets of previous editions are re-used and updated with new talks added to the TED repository in the meanwhile.

### Example of data format (tokens are joined via space)
# Source:
# ['Bakterien haben also nur sehr wenige Gene und genetische Informationen um sämtliche Merkmale , die sie ausführen , zu <unk> .',
#  'Die Idee von Krankenhäusern und Kliniken stammt aus den 1780ern . Es wird Zeit , dass wir unser Denken aktualisieren .',
#  'Ein Tier benötigt nur zwei Hundertstel einer Sekunde , um den Geruch zu unterscheiden , es geht also sehr schnell .',
#  'Es stellte sich heraus , dass die Ölkatastrophe eine weißes Thema war , dass <unk> eine vorherrschend schwarzes Thema war .',
#  'Wie ich in meinem Buch schreibe , bin ich genau so jüdisch , wie " Olive Garden " italienisch ist .',
#  'Es gibt einen belüfteten Ziegel den ich letztes Jahr in <unk> machte , als Konzept für New <unk> in Architektur .',
#  'Aber um die Zukunft des Wachstums zu verstehen , müssen wir Vorhersagen über die zugrunde liegenden <unk> des Wachstums machen .',
#  'Ich hatte einen Plan , und ich hätte nie gedacht , wem dabei eine Schlüsselrolle zukommen würde : dem Banjo .',
#  'Im Jahr 2000 hat er entdeckt , dass Ruß wahrscheinlich die zweitgrößte Ursache der globalen Erwärmung ist , nach CO2 .']
#
# Target:
# ['<s> They have very few genes , and genetic information to encode all of the traits that they carry out . </s>',
#  '<s> Humans invented the idea of hospitals and clinics in the 1780s . It is time to update our thinking . </s>',
#  '<s> An animal only needs two hundredths of a second to discriminate the scent , so it goes extremely fast . </s>',
#  '<s> It turns out that oil spill is a mostly white conversation , that cookout is a mostly black conversation . </s>',
#  "<s> As I say in my book , I 'm Jewish in the same way the Olive Garden is Italian . </s>",
#  "<s> There 's an aerated brick I did in <unk> last year , in Concepts for New Ceramics in Architecture . </s>",
#  '<s> but to understand the future of growth , we need to make predictions about the underlying drivers of growth . </s>',
#  '<s> I had a plan , and I never ever thought it would have anything to do with the banjo . </s>',
#  '<s> In 2000 , he discovered that soot was probably the second leading cause of global warming , after CO2 . </s>']

@torch.no_grad()
def pretrained_IWSLT_demo():
    """
    Demo on  IWSLT German-English Translation task
    """

    DECODING = "beam_search"
    BEAM_SIZE = 5
    MAX_DECODING_LEN = 60
    BATCH_SIZE = 100

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # don't forget to download spacy models
    # python -m spacy download de
    # python -m spacy download en

    print("Loading spacy models...")
    # spacy de tokenizer is very slow together with pdb
    if not os.path.isfile("spacy_de.pkl"):
        spacy_de = spacy.load('de')
        with open("spacy_de.pkl", "wb") as f:
            pickle.dump(spacy_de, f)
    else: #it seems to be faster to load pickled version of tokenizer
        with open("spacy_de.pkl", "rb") as f:
            spacy_de = pickle.load(f)
    spacy_en = spacy.load('en')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    # you may try to tokenize de like this to debug the code
    # but don't expect predictions to be meaningful, as the model has been trained via spacy tokenization
    # def tokenize_de(text):
    #     return str.split(text)

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"
    SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
    TGT = data.Field(tokenize=tokenize_en, init_token=BOS_WORD,
                     eos_token=EOS_WORD, pad_token=BLANK_WORD)
    INPUT_MAX_LEN = 100
    print("Loading dataset...")
    train, val, test = datasets.IWSLT.splits(
        exts=('.de', '.en'), fields=(SRC, TGT),
        filter_pred=lambda x: len(vars(x)['src']) <= INPUT_MAX_LEN and
                              len(vars(x)['trg']) <= INPUT_MAX_LEN)
    MIN_FREQ = 2
    print("Building vocabularies...")
    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train.trg, min_freq=MIN_FREQ)
    valid_iter = DataIterator(val, batch_size=BATCH_SIZE, device=device,
                              repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                              batch_size_fn=batch_size_fn, train=False)

    # using the pre_trained model from https://s3.amazonaws.com/opennmt-models/iwslt.pt
    if not os.path.exists("iwslt.pt"):
        wget.download("https://s3.amazonaws.com/opennmt-models/iwslt.pt")

    model = torch.load("iwslt.pt")

    for i, batch in enumerate(valid_iter):
        src = batch.src.transpose(0, 1)[:1]
        src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
        # Greedy decoding
        if DECODING == "greedy":
            # Greedy Decoding
            out = greedy_decode(model, src, src_mask, max_len=MAX_DECODING_LEN, start_symbol=TGT.vocab.stoi["<s>"])
        elif DECODING == "beam_search":
            # Beam Search
            decoded_hypotheses, logprobs = beam_search(model=model,
                                                       src=src,
                                                       src_mask=src_mask,
                                                       max_len=MAX_DECODING_LEN,
                                                       pad=TGT.vocab.stoi["<blank>"],
                                                       bos=TGT.vocab.stoi[BOS_WORD],
                                                       eos=TGT.vocab.stoi[EOS_WORD],
                                                       beam_size=BEAM_SIZE,
                                                       device=device)
            out = [h[0] for h in decoded_hypotheses]  # pick the most probable hypotheses for each element of batch
            out = torch.LongTensor(out)
        print("Source:", end="\t")
        for i in range(0, src.size(1)):
            sym = SRC.vocab.itos[src[0, i]]
            print(sym, end=" ")
        print()

        print("Translation:", end="\t")
        for i in range(0, out.size(1)):
            sym = TGT.vocab.itos[out[0, i]]
            if sym == "</s>": break
            print(sym, end=" ")
        print()
        print("Target:", end="\t")
        for i in range(1, batch.trg.size(0)):
            sym = TGT.vocab.itos[batch.trg.data[i, 0]]
            if sym == "</s>": break
            print(sym, end=" ")
        print()


if __name__ == "__main__":
    pretrained_IWSLT_demo()
