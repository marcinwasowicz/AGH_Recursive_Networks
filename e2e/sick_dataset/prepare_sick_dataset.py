import os

import os
import glob


def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)


def dependency_parse(filepath, cp="", tokenize=True):
    print("\nDependency parsing " + filepath)
    dirpath = os.path.dirname(filepath)
    filepre = os.path.splitext(os.path.basename(filepath))[0]
    tokpath = os.path.join(dirpath, filepre + ".toks")
    parentpath = os.path.join(dirpath, filepre + ".parents")
    relpath = os.path.join(dirpath, filepre + ".rels")
    tokenize_flag = "-tokenize - " if tokenize else ""
    cmd = (
        "java -cp %s DependencyParse -tokpath %s -parentpath %s -relpath %s %s < %s"
        % (cp, tokpath, parentpath, relpath, tokenize_flag, filepath)
    )
    os.system(cmd)


def constituency_parse(filepath, cp="", tokenize=True):
    dirpath = os.path.dirname(filepath)
    filepre = os.path.splitext(os.path.basename(filepath))[0]
    tokpath = os.path.join(dirpath, filepre + ".toks")
    parentpath = os.path.join(dirpath, filepre + ".cparents")
    tokenize_flag = "-tokenize - " if tokenize else ""
    cmd = "java -cp %s ConstituencyParse -tokpath %s -parentpath %s %s < %s" % (
        cp,
        tokpath,
        parentpath,
        tokenize_flag,
        filepath,
    )
    os.system(cmd)


def build_vocab(filepaths, dst_path, lowercase=True):
    vocab = set()
    for filepath in filepaths:
        with open(filepath) as f:
            for line in f:
                if lowercase:
                    line = line.lower()
                vocab |= set(line.split())
    with open(dst_path, "w") as f:
        for w in sorted(vocab):
            f.write(w + "\n")


def split(filepath, dst_dir):
    with open(filepath) as datafile, open(
        os.path.join(dst_dir, "a.txt"), "w"
    ) as afile, open(os.path.join(dst_dir, "b.txt"), "w") as bfile, open(
        os.path.join(dst_dir, "id.txt"), "w"
    ) as idfile, open(
        os.path.join(dst_dir, "sim.txt"), "w"
    ) as simfile:
        datafile.readline()
        for line in datafile:
            i, a, b, sim, ent = line.strip().split("\t")
            idfile.write(i + "\n")
            afile.write(a + "\n")
            bfile.write(b + "\n")
            simfile.write(sim + "\n")


def parse(dirpath, cp=""):
    dependency_parse(os.path.join(dirpath, "a.txt"), cp=cp, tokenize=True)
    dependency_parse(os.path.join(dirpath, "b.txt"), cp=cp, tokenize=True)
    constituency_parse(os.path.join(dirpath, "a.txt"), cp=cp, tokenize=True)
    constituency_parse(os.path.join(dirpath, "b.txt"), cp=cp, tokenize=True)


if __name__ == "__main__":
    compile_stanford_parsers_cmd = "javac -cp java:java/stanford-parser/stanford-parser.jar:java/stanford-parser/stanford-parser-3.5.1-models.jar java/*.java"
    os.system(compile_stanford_parsers_cmd)

    print("=" * 80)
    print("Preprocessing SICK dataset")
    print("=" * 80)

    base_dir = "./"
    data_dir = os.path.join(base_dir, "data")
    sick_dir = os.path.join(data_dir, "sick")
    java_dir = os.path.join(base_dir, "java")
    train_dir = os.path.join(sick_dir, "train")
    dev_dir = os.path.join(sick_dir, "dev")
    test_dir = os.path.join(sick_dir, "test")
    make_dirs([train_dir, dev_dir, test_dir])

    # java classpath for calling Stanford parser
    classpath = ":".join(
        [
            java_dir,
            os.path.join(java_dir, "stanford-parser/stanford-parser.jar"),
            os.path.join(java_dir, "stanford-parser/stanford-parser-3.5.1-models.jar"),
        ]
    )

    # split into separate files
    split(os.path.join(sick_dir, "SICK_train.txt"), train_dir)
    split(os.path.join(sick_dir, "SICK_trial.txt"), dev_dir)
    split(os.path.join(sick_dir, "SICK_test_annotated.txt"), test_dir)

    # parse sentences
    parse(train_dir, cp=classpath)
    parse(dev_dir, cp=classpath)
    parse(test_dir, cp=classpath)

    # get vocabulary
    build_vocab(
        glob.glob(os.path.join(sick_dir, "*/*.toks")),
        os.path.join(sick_dir, "vocab.txt"),
    )
    build_vocab(
        glob.glob(os.path.join(sick_dir, "*/*.toks")),
        os.path.join(sick_dir, "vocab-cased.txt"),
        lowercase=False,
    )
