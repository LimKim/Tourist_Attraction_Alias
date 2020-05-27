import sys
sent_end_sign = ['。', '！']


def cut_sent(sent_list, tag_seq):
    lc = len(sent_list)
    cut_result = []
    end_index = []
    for index in range(lc):
        if sent_list[index] in sent_end_sign:
            end_index.append(index+1)
    end_index = [0] + end_index
    if lc not in end_index:
        end_index.append(lc)
    for index in range(len(end_index)-1):
        cut_result.append([sent_list[end_index[index]:end_index[index+1]],
                           tag_seq[end_index[index]:end_index[index+1]]])
    return cut_result


def _process(filename, targetfile):
    with open(filename, 'r', encoding='utf8') as fr, open(targetfile, 'w', encoding='utf8') as fw:
        for line in fr:
            ll = eval(line)
            cut_result = cut_sent(ll[1], ll[2])
            for ssi in cut_result:
                fw.write(' '.join(ssi[0]) + '\n')


if __name__ == '__main__':
    _process('label.result.pro.test', 'laiye.corpus')
