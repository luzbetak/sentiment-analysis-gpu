def generate_balanced_dataset():
    sentiments = {
        # Negative examples with more nuanced negative sentiment
        0: [
            "This movie was absolutely terrible, a complete waste of time.",
            "The plot made no sense and the acting was atrocious.",
            "Despite the talented cast, the film fails to deliver.",
            "A disappointing effort that never finds its footing.",
            "The dialogue feels forced and unnatural throughout.",
            "Poor pacing makes this film a tedious experience.",
            "The story falls apart in the second half.",
            "Great potential wasted by poor execution.",
            "Somehow manages to get worse as it goes along.",
            "A mess of ideas that never comes together.",
            "The ending ruins everything that came before.",
            "Lacks any sense of coherence or purpose.",
            "Amateur filmmaking at its worst.",
            "Not even the special effects can save this disaster.",
            "A complete misfire on every level.",
            "Wastes its promising premise entirely.",
            "Incredibly disappointing given the source material.",
            "Makes all the wrong creative choices.",
            "A frustrating and confusing experience.",
            "Should have remained in development."
        ],
        # Neutral examples with more subtle and balanced opinions
        1: [
            "Has both strong points and weak moments.",
            "A mixed bag with some interesting elements.",
            "Decent enough but nothing special.",
            "The performances elevate average material.",
            "Neither particularly good nor bad.",
            "Some good ideas that aren't fully realized.",
            "Watchable but ultimately forgettable.",
            "Meets the basic requirements but little more.",
            "Could have been better, could have been worse.",
            "The director shows promise despite flaws.",
            "An acceptable way to pass the time.",
            "Somewhat entertaining but not memorable.",
            "Middle-of-the-road entertainment.",
            "Has its moments but doesn't quite gel.",
            "A fair attempt that falls short of its goals.",
            "Worth watching once but won't leave an impression.",
            "Technically competent but emotionally flat.",
            "The kind of film that divides audiences.",
            "Shows potential but needs refinement.",
            "An ambitious attempt with mixed results."
        ],
        # Positive examples with more varied positive sentiment
        2: [
            "A well-crafted film that exceeds expectations.",
            "Succeeds on multiple levels.",
            "Impressive execution of a challenging concept.",
            "The performances bring depth to every scene.",
            "A refreshing take on familiar themes.",
            "Keeps you engaged throughout.",
            "Thoughtful and well-executed storytelling.",
            "Strong direction elevates the material.",
            "A pleasant surprise that delivers.",
            "Handles complex themes with grace.",
            "Memorable characters and compelling plot.",
            "A solid addition to the genre.",
            "Demonstrates real creative vision.",
            "Worth multiple viewings.",
            "Shows remarkable attention to detail.",
            "A satisfying blend of style and substance.",
            "Consistently entertaining throughout.",
            "Lives up to the high expectations.",
            "Brings something new to the table.",
            "A genuinely enjoyable experience."
        ]
    }

    text_data = []
    labels = []

    for sentiment, examples in sentiments.items():
        text_data.extend(examples)
        labels.extend([sentiment] * len(examples))

    return text_data, labels

def get_test_examples():
    return [
        # More nuanced test examples
        "While the visual effects impress, the narrative struggles to maintain coherence.",
        "A solid effort that manages to entertain despite familiar elements.",
        "The strong cast elevates this otherwise standard genre piece.",
        "Interesting ideas and decent execution make for a watchable experience.",
        "Shows promise but doesn't quite reach its full potential.",
        "The director's ambition is evident, though not every risk pays off.",
        "A competent production that neither excels nor disappoints.",
        "Moments of brilliance mixed with occasional missteps.",
        "Worth watching for the performances, even if the story meanders.",
        "An admirable attempt that partially succeeds in its goals."
    ]
    
if __name__ == "__main__":
    # Test the data generation
    text_data, labels = generate_balanced_dataset()
    print(f"Total examples: {len(text_data)}")
    print(f"Examples per class: {len(text_data) // 3}")
    # Print a few examples from each class
    for i in range(3):
        print(f"\nClass {i} examples:")
        class_examples = [text for text, label in zip(text_data, labels) if label == i][:3]
        for example in class_examples:
            print(f"- {example}")
