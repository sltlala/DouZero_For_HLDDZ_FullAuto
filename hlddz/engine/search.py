import time
from douzero.env.move_generator import MovesGener
from douzero.env.move_detector import get_move_type
from douzero.env import move_selector

EnvCard2RealCard = {3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
                    8: '8', 9: '9', 10: 'T', 11: 'J', 12: 'Q',
                    13: 'K', 14: 'A', 17: '2', 20: 'X', 30: 'D'}


def action_to_str(action):
    if len(action) == 0:
        return "Pass"
    else:
        return "".join([EnvCard2RealCard[card] for card in action])


def type_exist(mlist, type):
    if not isinstance(mlist, list):
        return False
    for item in mlist:
        if not isinstance(item, type):
            return False
    return True


def action_in_tree(path_list, action):
    for ac in path_list:
        ac[0].sort()
        if action == ac[0]:
            return ac
    return None


def search_actions(my_cards, other_cards, path_list, rival_move=None, prev_moves=None):
    """枚举所有「对手压不住、能一路把牌打完」的出牌顺序，结果原地追加进 path_list。

    返回 path_list 本身。
    历史问题（已修复）：旧版末尾无条件 `return None`，导致
      1) 函数只能靠修改传入的 path_list 这个副作用传结果，极易误用；
      2) 递归里的 `if actions is not None ...` 恒为假，整段分支是死代码。
    path_list 超过 100 条时提前停止搜索。
    """
    if len(path_list) > 100:
        return path_list
    if prev_moves is None:
        my_cards.sort()
        other_cards.sort()
    my_gener = MovesGener(my_cards)
    other_gener = MovesGener(other_cards)
    other_bombs = other_gener.gen_type_4_bomb()
    other_bombs.extend(other_gener.gen_type_5_king_bomb())
    rival_move_info = {}
    type_range = [4, 5, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    if rival_move is not None:
        if len(rival_move) > 0:
            rival_move_info = get_move_type(rival_move)
            type_range = [4, 5, rival_move_info["type"]]
        else:
            rival_move = None

    for mtype in type_range:
        my_moves = my_gener.gen_moves_by_type(mtype)
        if len(my_moves) == 0:
            continue
        if mtype == 4:
            other_moves = other_bombs
        else:
            other_moves = other_gener.gen_moves_by_type(mtype)
        for move in my_moves:
            if len(move) != len(my_cards):
                if mtype != 4 and mtype != 5 and len(other_bombs) > 0:
                    break
                if len(move_selector.filter_type_n(mtype, other_moves, move)) == 0:
                    if rival_move is not None:
                        move_info = get_move_type(move)
                        if mtype != 5:
                            if "rank" in move_info and "rank" in rival_move_info and move_info["rank"] <= \
                                    rival_move_info["rank"]:
                                continue
                            if "len" in move_info and move_info["len"] != rival_move_info["len"]:
                                continue
                            if rival_move_info["type"] == 5:
                                continue

                    new_cards = my_cards.copy()
                    for card in move:
                        new_cards.remove(card)
                    if prev_moves is not None:
                        new_prev = prev_moves.copy()
                        new_prev.append(move)
                    else:
                        new_prev = [move]
                    # 递归只负责继续向 path_list 追加更长的完整路径，返回值不再使用
                    search_actions(new_cards, other_cards, path_list, prev_moves=new_prev)
                    del new_prev
                    del new_cards
            else:
                if rival_move is not None:
                    move_info = get_move_type(move)
                    if "rank" in move_info and "rank" in rival_move_info and move_info["rank"] <= rival_move_info[
                        "rank"]:
                        continue
                    if "len" in move_info and move_info["len"] != rival_move_info["len"]:
                        continue
                    if rival_move_info["type"] == 5:
                        continue
                if prev_moves is not None:
                    new_path = prev_moves.copy()
                    new_path.append(move)
                    path_list.append(new_path)
                else:
                    path_list.append([move])
    del my_gener, other_gener, other_bombs, my_cards, other_cards
    return path_list


def eval_path(path):
    bomb = 0
    for action in path:
        if 30 in action and 20 in action or len(action) == 4 and len(set(action)) == 1:
            bomb += 1
    return 1 + bomb - len(path) * 0.05


def select_optimal_path(path_list):
    if len(path_list) != 0:
        max_path = max(path_list, key=lambda x: eval_path(x))
        for action in max_path:
            action.sort()
        return max_path
    else:
        return None


def check_42(path):
    for action in path:
        move_type = get_move_type(action)
        if move_type["type"] == 13 or move_type["type"] == 14:
            return True
    return False


if __name__ == "__main__":
    my_cards = [20, 30, 10, 10]
    other_cards = [4, 13, 5, 5, 5, 5, 8, 8, 8, 8, 12, 12, 13]
    st = time.time()
    paths = []
    search_actions(my_cards, other_cards, paths, rival_move=[13])
    print(time.time() - st)
    # print(paths)
    for path in paths:
        print(path)
    print(len(paths))
    path = select_optimal_path(paths)
    print("optimal", path)
