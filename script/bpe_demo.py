"""
BPE (Byte Pair Encoding) 最简实现
================================
核心思想：从「每个字符一个 token」开始，反复找出语料中
出现频率最高的相邻符号对，把它合并成一个新 token，
重复 N 次得到词表。高频词会慢慢合并成整词，低频词保留在字符级。

运行：python bpe_demo.py
"""
from collections import Counter


def get_stats(vocab):
    """统计所有单词中相邻符号对的频率

    Args:
        vocab: {tuple(符号): 出现次数}，如 {('l','o','w','</w>'): 5}

    Returns:
        Counter: {(符号a, 符号b): 频率}
    """
    pairs = Counter()
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq
    return pairs


def merge_pair(pair, vocab):
    """把词表中所有出现的 pair 合并成一个符号

    Args:
        pair: 待合并的符号对，如 ('e', 's')
        vocab: 当前词表 {tuple(符号): 出现次数}

    Returns:
        dict: 合并后的新词表
    """
    new_vocab = {}
    bigram = ''.join(pair)
    for word, freq in vocab.items():
        new_word = []
        i = 0
        while i < len(word):
            # 找到 pair 就合并
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(bigram)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_vocab[tuple(new_word)] = freq
    return new_vocab


def train_bpe(corpus, num_merges=10, verbose=True):
    """训练 BPE

    Args:
        corpus: 单词列表
        num_merges: 合并次数（即新增 token 数）
        verbose: 是否打印每步合并结果

    Returns:
        merge_rules: 按顺序记录的合并规则，编码时要按此顺序应用
    """
    # 初始词表：每个词拆成字符 + 词尾标记 </w>
    # 用 Counter 统计词频，重复出现的单词按真实次数累加
    vocab = Counter(tuple(w) + ('</w>',) for w in corpus)
    merge_rules = []

    for step in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)  # 频率最高的符号对
        vocab = merge_pair(best, vocab)
        merge_rules.append(best)
        if verbose:
            sample = [list(k) for k in list(vocab.keys())[:3]]
            print(f"step {step}: 合并 {best} (频率={pairs[best]}) -> {sample}...")
    return merge_rules


def encode(word, merge_rules):
    """按训练时的合并顺序切分一个新单词（简化版实现）

    注：工业实现（HuggingFace tokenizers / tiktoken）会用
    优先级队列或 Trie 树加速，这里为可读性按规则顺序逐个应用。

    Args:
        word: 待切分的字符串
        merge_rules: 训练得到的合并规则列表

    Returns:
        list: 切分后的 token 列表
    """
    word = tuple(word) + ('</w>',)
    for pair in merge_rules:
        bigram = ''.join(pair)
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(bigram)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        word = tuple(new_word)
    return list(word)


def decode(tokens):
    """去掉词尾标记后拼接还原文本

    Args:
        tokens: encode 输出的 token 列表

    Returns:
        str: 还原后的文本
    """
    return ''.join(tokens).replace('</w>', ' ').strip()


if __name__ == "__main__":
    # ---------- 1. 训练 ----------
    corpus = ["low", "low", "low", "low", "low",
              "lower", "lower", "newest", "newest",
              "newest", "newest", "newest", "widest", "widest"]

    merge_rules = train_bpe(corpus, num_merges=10)

    # ---------- 2. 编码 / 解码测试 ----------
    print("\n学到的合并规则:", merge_rules)
    print("编码 'lowest' :", encode("lowest", merge_rules))  # 训练集中没见过
    print("编码 'newest' :", encode("newest", merge_rules))
    print("解码还原     :", decode(encode("lowest", merge_rules)))

    # ---------- 3. 生产环境用法（可选） ----------
    # pip install tiktoken
    # import tiktoken
    # enc = tiktoken.get_encoding("gpt2")
    # print(enc.encode("lowest"))      # [11749, 279] -> 两个 token
    # print(enc.decode([11749, 279]))  # 'lowest'
