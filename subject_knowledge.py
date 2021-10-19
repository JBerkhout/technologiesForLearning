import pandas as pd
from pre_processing import get_topics_names_dict

# Booleans are used to indicate whether a student has some prior knowledge on a subject or not, meaning
# whether or not they presented a topic in the same theme.

def compute_subject_knowledge():
    data_dict = pd.read_excel("data_v2.xlsx", None)
    df = data_dict['main']
    output = []

    for student in range(1, 45):
        student_row = df.loc[df['User'] == student]
        theme_presented = str(student_row['Topic'].tolist()[0])[0]

        new_dict = {"student": student}

        topic_names = get_topics_names_dict()
        for topic in range(1, 23):
            knowledge = (theme_presented == str(topic_names[topic])[0])
            new_dict['topic'+str(topic)] = knowledge

        output.append(new_dict)

    return output


def save_subject_knowledge_excel():
    out = compute_subject_knowledge()
    df = pd.DataFrame.from_dict(out)
    df.to_excel('subject_knowledge.xlsx')


# MAIN
# save_subject_knowledge_excel()