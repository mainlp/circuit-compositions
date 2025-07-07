"""
Unit tests for manual string function mappings
"""

from comp_rep.eval.cross_task_evaluations import src_to_func_target


def test_copy_to_copy_mapping():
    """
    Tests the functionality of src_to_func_target when mapping 'copy' to 'copy'.
    """
    data_func_name = "copy"
    new_func_name = "copy"

    src_str_lists = [
        "copy copy W1 K1 E1 N1 D1".split(" "),
        "copy X1 U1 V1 Z1 W1".split(" "),
        "copy B1 Q1 D1 Q1 I1".split(" "),
        "copy copy copy copy K1 W1 D1 D1 V1".split(" "),
    ]
    target_str_lists = [
        "W1 K1 E1 N1 D1".split(" "),
        "X1 U1 V1 Z1 W1".split(" "),
        "B1 Q1 D1 Q1 I1".split(" "),
        "K1 W1 D1 D1 V1".split(" "),
    ]

    for src_str_list, target_str_list in zip(src_str_lists, target_str_lists):
        mapped_str_list = src_to_func_target(
            src_str_list=src_str_list,
            data_func_name=data_func_name,
            new_func_name=new_func_name,
        )
        assert mapped_str_list == target_str_list


def test_copy_to_reverse_mapping():
    """
    Tests the functionality of src_to_func_target when mapping 'copy' to 'reverse'.
    """
    data_func_name = "copy"
    new_func_name = "reverse"

    src_str_lists = [
        "copy copy W1 K1 E1 N1 D1".split(" "),
        "copy X1 U1 V1 Z1 W1".split(" "),
        "copy B1 Q1 D1 Q1 I1".split(" "),
        "copy copy copy copy K1 W1 D1 D1 V1".split(" "),
    ]
    target_str_lists = [
        "W1 K1 E1 N1 D1".split(" "),
        "W1 Z1 V1 U1 X1".split(" "),
        "I1 Q1 D1 Q1 B1".split(" "),
        "K1 W1 D1 D1 V1".split(" "),
    ]

    for src_str_list, target_str_list in zip(src_str_lists, target_str_lists):
        mapped_str_list = src_to_func_target(
            src_str_list=src_str_list,
            data_func_name=data_func_name,
            new_func_name=new_func_name,
        )
        assert mapped_str_list == target_str_list


def test_copy_to_echo_mapping():
    """
    Tests the functionality of src_to_func_target when mapping 'copy' to 'echo'.
    """
    data_func_name = "copy"
    new_func_name = "echo"

    src_str_lists = [
        "copy copy W1 K1 E1 N1 D1".split(" "),
        "copy X1 U1 V1 Z1 W1".split(" "),
        "copy B1 Q1 D1 Q1 I1".split(" "),
        "copy copy copy copy K1 W1 D1 D1 V1".split(" "),
    ]
    target_str_lists = [
        "W1 K1 E1 N1 D1 D1 D1".split(" "),
        "X1 U1 V1 Z1 W1 W1".split(" "),
        "B1 Q1 D1 Q1 I1 I1".split(" "),
        "K1 W1 D1 D1 V1 V1 V1 V1 V1".split(" "),
    ]

    for src_str_list, target_str_list in zip(src_str_lists, target_str_lists):
        mapped_str_list = src_to_func_target(
            src_str_list=src_str_list,
            data_func_name=data_func_name,
            new_func_name=new_func_name,
        )
        assert mapped_str_list == target_str_list


def test_repeat_to_reverse_mapping():
    """
    Tests the functionality of src_to_func_target when mapping 'repeat' to 'reverse'.
    """
    data_func_name = "repeat"
    new_func_name = "reverse"

    src_str_lists = [
        "repeat repeat W1 K1 E1 N1 D1".split(" "),
        "repeat X1 U1 V1 Z1 W1".split(" "),
        "repeat B1 Q1 D1 Q1 I1".split(" "),
        "repeat repeat repeat repeat K1 W1 D1 D1 V1".split(" "),
    ]
    target_str_lists = [
        "W1 K1 E1 N1 D1".split(" "),
        "W1 Z1 V1 U1 X1".split(" "),
        "I1 Q1 D1 Q1 B1".split(" "),
        "K1 W1 D1 D1 V1".split(" "),
    ]

    for src_str_list, target_str_list in zip(src_str_lists, target_str_lists):
        mapped_str_list = src_to_func_target(
            src_str_list=src_str_list,
            data_func_name=data_func_name,
            new_func_name=new_func_name,
        )
        assert mapped_str_list == target_str_list


def test_reverse_to_swap_first_last_mapping():
    """
    Tests the functionality of src_to_func_target when mapping 'reverse' to 'swap_first_last'.
    """
    data_func_name = "reverse"
    new_func_name = "swap_first_last"

    src_str_lists = [
        "reverse reverse W1 K1 E1 N1 D1".split(" "),
        "reverse X1 U1 V1 Z1 W1".split(" "),
        "reverse B1 Q1 D1 Q1 I1".split(" "),
        "reverse reverse reverse reverse K1 W1 D1 D1 V1".split(" "),
    ]
    target_str_lists = [
        "W1 K1 E1 N1 D1".split(" "),
        "W1 U1 V1 Z1 X1".split(" "),
        "I1 Q1 D1 Q1 B1".split(" "),
        "K1 W1 D1 D1 V1".split(" "),
    ]

    for src_str_list, target_str_list in zip(src_str_lists, target_str_lists):
        mapped_str_list = src_to_func_target(
            src_str_list=src_str_list,
            data_func_name=data_func_name,
            new_func_name=new_func_name,
        )
        assert mapped_str_list == target_str_list


def test_echo_to_copy_mapping():
    """
    Tests the functionality of src_to_func_target when mapping 'echo' to 'copy'.
    """
    data_func_name = "echo"
    new_func_name = "copy"

    src_str_lists = [
        "echo echo W1 K1 E1 N1 D1".split(" "),
        "echo X1 U1 V1 Z1 W1".split(" "),
        "echo B1 Q1 D1 Q1 I1".split(" "),
        "echo echo echo echo K1 W1 D1 D1 V1".split(" "),
    ]
    target_str_lists = [
        "W1 K1 E1 N1 D1".split(" "),
        "X1 U1 V1 Z1 W1".split(" "),
        "B1 Q1 D1 Q1 I1".split(" "),
        "K1 W1 D1 D1 V1".split(" "),
    ]

    for src_str_list, target_str_list in zip(src_str_lists, target_str_lists):
        mapped_str_list = src_to_func_target(
            src_str_list=src_str_list,
            data_func_name=data_func_name,
            new_func_name=new_func_name,
        )
        assert mapped_str_list == target_str_list


def test_append_to_append_mapping():
    """
    Tests the functionality of src_to_func_target when mapping 'append' to 'append'.
    """
    data_func_name = "append"
    new_func_name = "append"

    src_str_lists = [
        "append B1 R1 O1 R1 , I1 H1 M1 O1".split(" "),
        "append append append B1 T1 Z1 T1 , V1 W1 W1 F1 , A1 I1 T1 S1 Z1 , G1 H1 A1".split(
            " "
        ),
        "append E1 D1 F1 H1 , D1 L1 H1 U1".split(" "),
    ]
    target_str_lists = [
        "B1 R1 O1 R1 I1 H1 M1 O1".split(" "),
        "B1 T1 Z1 T1 V1 W1 W1 F1 A1 I1 T1 S1 Z1 G1 H1 A1".split(" "),
        "E1 D1 F1 H1 D1 L1 H1 U1".split(" "),
    ]

    for src_str_list, target_str_list in zip(src_str_lists, target_str_lists):
        mapped_str_list = src_to_func_target(
            src_str_list=src_str_list,
            data_func_name=data_func_name,
            new_func_name=new_func_name,
        )
        assert mapped_str_list == target_str_list


def test_append_to_prepend_mapping():
    """
    Tests the functionality of src_to_func_target when mapping 'append' to 'prepend'.
    """
    data_func_name = "append"
    new_func_name = "prepend"

    src_str_lists = [
        "append B1 R1 O1 R1 , I1 H1 M1 O1".split(" "),
        "append append append B1 T1 Z1 T1 , V1 W1 W1 F1 , A1 I1 T1 S1 Z1 , G1 H1 A1".split(
            " "
        ),
        "append E1 D1 F1 H1 , D1 L1 H1 U1".split(" "),
    ]
    target_str_lists = [
        "I1 H1 M1 O1 B1 R1 O1 R1".split(" "),
        "G1 H1 A1 A1 I1 T1 S1 Z1 V1 W1 W1 F1 B1 T1 Z1 T1".split(" "),
        "D1 L1 H1 U1 E1 D1 F1 H1".split(" "),
    ]

    for src_str_list, target_str_list in zip(src_str_lists, target_str_lists):
        mapped_str_list = src_to_func_target(
            src_str_list=src_str_list,
            data_func_name=data_func_name,
            new_func_name=new_func_name,
        )
        assert mapped_str_list == target_str_list


def test_prepend_to_remove_second_mapping():
    """
    Tests the functionality of src_to_func_target when mapping 'prepend' to 'remove_second'.
    """
    data_func_name = "prepend"
    new_func_name = "remove_second"

    src_str_lists = [
        "prepend B1 R1 O1 R1 , I1 H1 M1 O1".split(" "),
        "prepend prepend prepend B1 T1 Z1 T1 , V1 W1 W1 F1 , A1 I1 T1 S1 Z1 , G1 H1 A1".split(
            " "
        ),
        "prepend E1 D1 F1 H1 , D1 L1 H1 U1".split(" "),
    ]
    target_str_lists = [
        "B1 R1 O1 R1".split(" "),
        "B1 T1 Z1 T1".split(" "),
        "E1 D1 F1 H1".split(" "),
    ]

    for src_str_list, target_str_list in zip(src_str_lists, target_str_lists):
        mapped_str_list = src_to_func_target(
            src_str_list=src_str_list,
            data_func_name=data_func_name,
            new_func_name=new_func_name,
        )
        assert mapped_str_list == target_str_list


def test_remove_second_to_remove_first_mapping():
    """
    Tests the functionality of src_to_func_target when mapping 'remove_second' to 'remove_first'.
    """
    data_func_name = "remove_second"
    new_func_name = "remove_first"

    src_str_lists = [
        "remove_second B1 R1 O1 R1 , I1 H1 M1 O1".split(" "),
        "remove_second remove_second remove_second B1 T1 Z1 T1 , V1 W1 W1 F1 , A1 I1 T1 S1 Z1 , G1 H1 A1".split(
            " "
        ),
        "remove_second E1 D1 F1 H1 , D1 L1 H1 U1".split(" "),
    ]
    target_str_lists = [
        "I1 H1 M1 O1".split(" "),
        "G1 H1 A1".split(" "),
        "D1 L1 H1 U1".split(" "),
    ]

    for src_str_list, target_str_list in zip(src_str_lists, target_str_lists):
        mapped_str_list = src_to_func_target(
            src_str_list=src_str_list,
            data_func_name=data_func_name,
            new_func_name=new_func_name,
        )
        assert mapped_str_list == target_str_list
