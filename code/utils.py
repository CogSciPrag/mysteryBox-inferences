import random

def format_item(r, item_template):
    """
    Helper function for formatting the single item information into
    intructions template. Return text up to the trigger word to be scored.
    """
    print("R in format item ", r)
    answer_options = [r["Answer_option1"], r["Answer_option2"]]
    random.shuffle(answer_options)
    item_text = item_template.format(
        character=r["Character"],
        context_box1=r["Content_Box1"],
        context_box2=r["Content_Box2"],
        context_box3=r["Content_Box3"],
        sentence=r["Sentence"],
        answer_option1=answer_options[0],
        answer_option2=answer_options[1],
    )

    return item_text