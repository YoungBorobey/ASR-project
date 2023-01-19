import editdistance
# Don't forget to support cases when target_text == ''

def calc_cer(target_text, predicted_text) -> float:
    # TODO: your code here
    if len(target_text) == 0:
        return 1
    return editdistance.distance(target_text, predicted_text) / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    # TODO: your code here
    splitted_target = target_text.split(' ')
    if len(splitted_target) == 0:
        return 1
    splitted_pred = predicted_text.split(' ')
    # res = editdistance.distance(splitted_target, splitted_pred) / len(splitted_target)
    # print('target:', target_text)
    # print('pred:', predicted_text)
    # print('wer:', res)
    # print(len(target_text), len(predicted_text))
    return editdistance.distance(splitted_target, splitted_pred) / len(splitted_target)