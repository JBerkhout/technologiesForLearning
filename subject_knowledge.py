import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pre_processing import get_topics_names_dict, get_topic_presenters_dict


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
            new_dict['topic' + str(topic)] = knowledge

        output.append(new_dict)

    return output


def save_subject_knowledge_excel():
    out = compute_subject_knowledge()
    df = pd.DataFrame.from_dict(out)
    df.to_excel('subject_knowledge.xlsx')


def get_theme_presenters(cur_topic):
    users_with_knowledge = []

    if cur_topic <= 6:
        theme_topics = range(1, 7)
    elif cur_topic <= 9:
        theme_topics = range(7, 10)
    elif cur_topic <= 13:
        theme_topics = range(10, 14)
    elif cur_topic <= 16:
        theme_topics = range(14, 17)
    else:
        theme_topics = range(17, 23)

    for topic in theme_topics:
        users_with_knowledge += get_topic_presenters_dict()[topic]

    return users_with_knowledge


def read_topic_variability_statistics_topic_knowledge():
    data_dict = pd.read_excel("data_v2.xlsx", None)
    df_out_with, df_out_without, output_with, output_without = [], [], [], []

    for topic in range(1, 23):
        tab_name = 'topic' + str(topic)
        df = data_dict[tab_name]
        users_with_knowledge = get_theme_presenters(topic)
        df_with = df[df["User"].isin(users_with_knowledge)]
        df_without = df[~df["User"].isin(users_with_knowledge)]

        for rubric in range(1, 9):
            rubric_grades = df_with['Grade' + str(rubric)].tolist()
            new_dict = {"topic": topic,
                        "rubric": rubric,
                        "variance": np.var(rubric_grades),
                        }
            df_out_with.append(new_dict)

            rubric_grades = df_without['Grade' + str(rubric)].tolist()
            new_dict = {"topic": topic,
                        "rubric": rubric,
                        "variance": np.var(rubric_grades),
                        }
            df_out_without.append(new_dict)

        df_with = pd.DataFrame(df_out_with)
        df_without = pd.DataFrame(df_out_without)

        topic_selection = df_with.loc[df_with['topic'] == topic]
        variance_list = topic_selection['variance'].tolist()
        mean_variance = np.mean(variance_list)

        # Now let's save all this information per topic in a dictionary.
        topic_variability = {
            "topic": topic,
            "mean_variance": mean_variance}
        output_with.append(topic_variability)

        if pd.isna(topic_variability["mean_variance"]):
            topic_variability["mean_variance"] = 0

        topic_selection = df_without.loc[df_with['topic'] == topic]
        variance_list = topic_selection['variance'].tolist()
        mean_variance = np.mean(variance_list)

        # Now let's save all this information per topic in a dictionary.
        topic_variability = {
            "topic": topic,
            "mean_variance": mean_variance}

        if pd.isna(topic_variability["mean_variance"]):
            topic_variability["mean_variance"] = 0

        output_without.append(topic_variability)

    # print(output_with)
    # print(output_without)
    return output_with, output_without


def plot_topic_variability_subject_knowledge():
    out_with, out_without = read_topic_variability_statistics_topic_knowledge()
    df_with = pd.DataFrame.from_dict(out_with)
    df_without = pd.DataFrame.from_dict(out_without)
    topic_names = df_with.topic.map(get_topics_names_dict())

    x_pos = np.arange(len(topic_names))
    width = 0.35

    # Plotting the multiple bar graphs on the same figure
    plt.bar(x_pos, df_without.mean_variance, color='r', width=width, label='Without subject knowledge')
    plt.bar(x_pos + width, df_with.mean_variance, color='y', width=width, label='With subject knowledge')

    # plt.bar(topic_names, df.mean_variance)
    plt.title('Bar plot of variability per topic')
    plt.xticks(x_pos + width, topic_names, rotation='vertical')
    plt.xlabel('Topic')
    plt.ylabel('Variance')
    plt.legend()
    plt.show()


def plot_topic_variability_theme_grouped_subject_knowledge():
    out_with, out_without = read_topic_variability_statistics_topic_knowledge()
    df_with = pd.DataFrame(out_with)
    df_without = pd.DataFrame(out_without)
    topic_names = df_with.topic.map(get_topics_names_dict())

    # x_pos = np.arange(len(topic_names))
    # width = 0.35

    # (manually) specifying which topics belong to which theme
    y_t1 = df_with.mean_variance[0:6]
    y_t2 = df_with.mean_variance[6:9]
    y_t3 = df_with.mean_variance[9:13]
    y_t4 = df_with.mean_variance[13:16]
    y_t5 = df_with.mean_variance[16:22]

    y_o1 = df_without.mean_variance[0:6]
    y_o2 = df_without.mean_variance[6:9]
    y_o3 = df_without.mean_variance[9:13]
    y_o4 = df_without.mean_variance[13:16]
    y_o5 = df_without.mean_variance[16:22]

    # ensuring correct spacing between topics
    x_t1 = np.arange(len(y_t1))
    x_t2 = 1 + np.arange(len(y_t2)) + len(y_t1)
    x_t3 = 2 + np.arange(len(y_t3)) + len(y_t1) + len(y_t2)
    x_t4 = 3 + np.arange(len(y_t4)) + len(y_t1) + len(y_t2) + len(y_t3)
    x_t5 = 4 + np.arange(len(y_t5)) + len(y_t1) + len(y_t2) + len(y_t3) + len(y_t4)

    fig, ax = plt.subplots()
    ax.bar(x_t1, y_t1, color='r', alpha=0.5,  label="With subject knowledge")
    ax.bar(x_t2, y_t2, color='r', alpha=0.5)
    ax.bar(x_t3, y_t3, color='r', alpha=0.5)
    ax.bar(x_t4, y_t4, color='r', alpha=0.5)
    ax.bar(x_t5, y_t5, color='r', alpha=0.5)

    ax.bar(x_t1, y_o1, color='b', label="Without subject knowledge", alpha=0.5)
    ax.bar(x_t2, y_o2, color='b', alpha=0.5)
    ax.bar(x_t3, y_o3, color='b', alpha=0.5)
    ax.bar(x_t4, y_o4, color='b', alpha=0.5)
    ax.bar(x_t5, y_o5, color='b', alpha=0.5)

    ax.set_title('Bar plot of variability per topic grouped per theme')
    ax.set_ylabel('Variance')
    ax.set_xlabel('Topic')
    ax.set_xticks(np.concatenate((x_t1, x_t2, x_t3, x_t4, x_t5)))
    ax.set_xticklabels(topic_names, rotation='vertical')
    ax.legend()
    plt.show()



# MAIN
# save_subject_knowledge_excel()
