for ds in ["amazon", "yelp"]:
    if ds == "amazon":
        ori_ds = "amazon"
        ds = "book"
    else:
        ori_ds = "yelp"
        sp = -500
    base_dir = f"../../data/other_systems/drg/{ds}-drg"
    ori_train, tsf_train = [], []
    for sty in ["0", "1"]:
        if ds == "book":
            if sty == "0":
                sp = -4886
            else:
                sp = -4844
        ori_tt, tsf_tt = [], []
        tsf_sty = []
        file_name = f"{base_dir}/sentiment.test.{sty}.DeleteAndRetrieve.{ori_ds}"
        with open(file_name, 'r', encoding="utf-8") as f:
            for line in f:
                ori, tsf = line.strip().split("\t")[:2]
                ori_tt.append(ori)
                tsf_tt.append(tsf)
        ori_train += ori_tt[:sp]
        tsf_train += tsf_tt[:sp]
        with open(f"{base_dir}/style.train.{sty}.tsf", 'w+', encoding='utf-8') as f:
            for line in tsf_tt[:sp]:
                f.write(line + "\n")
        with open(f"{base_dir}/style.test.{sty}.tsf", 'w+', encoding='utf-8') as f:
            for line in tsf_tt[sp:]:
                f.write(line + "\n")
        with open(f"{base_dir}/style.test.{sty}", 'w+', encoding='utf-8') as f:
            for line in ori_tt[sp:]:
                f.write(line + "\n")
    with open(f"{base_dir}/{ds}-drg.train.ori", "w+", encoding='utf-8') as f:
        for line in ori_train:
            f.write(line + "\n")
    with open(f"{base_dir}/{ds}-drg.train.tsf", "w+", encoding='utf-8') as f:
        for line in tsf_train:
            f.write(line + "\n")