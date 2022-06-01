from logging import raiseExceptions
import sys
import stanza

DATA_DIR_ENG='/home/ania/Documents/NLP_token/data/eng_data.conl'
DATA_DIR_IT='/home/ania/Documents/NLP_token/data/it_data.conl'

def read_data(data_dir):
    group_sent = ''
    group_tok_sent = []
    tokenized_single_sent = []
    single_sent = ''
    input_data = []
    target_data = []

    with open(data_dir) as f:
        lines = f.readlines()
        i = 0
        for line in lines:
            tmp = line.split()

            if line == "\n":
                group_sent += single_sent
                group_tok_sent.append(tokenized_single_sent)
                tokenized_single_sent = []
                single_sent = ''
                i  += 1

                if i % 2 == 0:
                    input_data.append(group_sent)
                    target_data.append(group_tok_sent)

                    group_sent = ''
                    group_tok_sent = []  
            else:
                single_sent = single_sent + tmp[1] + ' '
                tokenized_single_sent.append(tmp[1])

    return input_data, target_data


def tokenize(lanuage, data_dir):
    input_data, target_data = read_data(data_dir)
    stanza.download(lanuage)
    nlp = stanza.Pipeline(lanuage, use_gpu=False)

    accuracy = 0
    num_accuracy = 0

    for i in range(len(input_data)):
        doc = nlp(input_data[i])

        tmp = 0
        for sentence in doc.sentences:
            for j in range(len(sentence.words)):
                num_accuracy += 1
                if tmp > 1 or len(target_data[i][tmp]) <= j:
                    continue
                if target_data[i][tmp][j] == sentence.words[j].text:
                    accuracy += 1
            tmp += 1

    return float(accuracy / num_accuracy)

if __name__ == '__main__':
    lang = sys.argv[1]
    if lang == 'en':
        data_dir = DATA_DIR_ENG
    elif lang =='it':
        data_dir = DATA_DIR_IT
    else:
        raiseExceptions("Wrong language. Set 'en' or 'it'.")
    accuracy = tokenize(lang, data_dir)
    print("Accuracy stanza: ")
    print(accuracy)
