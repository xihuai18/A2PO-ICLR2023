def args_str2bool(flag: str):
    assert flag == "True" or flag == "False"
    if flag == "True":
        return True
    else:
        return False