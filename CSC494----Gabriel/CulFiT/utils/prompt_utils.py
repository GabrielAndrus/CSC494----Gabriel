import json

def truncate_to_token_limit(prompt, max_tokens=12000):
    """
    Truncate the input prompt to ensure it is within the maximum token limit.
    """
    # Average length of one token is roughly 4 characters for English
    avg_token_size = 4
    max_chars = max_tokens * avg_token_size

    # Truncate the tokens if necessary
    if len(prompt) > max_chars:
        # Truncate and keep the most recent tokens
        truncated_prompt = prompt[-max_chars:]
        print(f"original length: {len(prompt)}")
        print(f"truncated length: {len(truncated_prompt)}")
        return truncated_prompt
    else:
        return prompt


FEEDBACK_PROMPT_TEMPLATE = '''You are an expert reviewer for a cultural knowledge question answering system. You have a plenty of cultural knowledge in {}, and your goal is to help the system to improve its performance. You are given the following question and answer. Please review the proposed answer and provide a paragraph of critique on its culturally-awareness, correctness, and comprehensiveness. Critique should be in the language that people in {} speak most and English.

Remember, if the most people in the above cultural group don't speak English, you should provide two critiques in the cultural group language and English, respectively. 

And importantly, if the cultural group mainly speak English or spans more than one language, you should provide the critiques in English only and do not repeat your crituqes.

Please be as helpful as possible and provide constructive critique.

Bear in mind that you always need to provide two critiques in total and each critiques should be no less than 100 tokens.

                
Question: {}
                                
Proposed answer: {}
                                
Feedback(s):[/INST]'''



KNOWLEDGE_PROMPT_SYS_TEMPLATE = '''You are an expert knowledge extractor for a cultural system. You have a plenty of cultural knowledge. You are provided with a piece of cultural knowledge or assertions extracted from wikipedia or online social media. Your task is to translate this piece of knowledge into a short, descriptive paragraph that is easy to understand for others. The cultural knowledge is encoded as JSON format, and below is its JSON fields:
{"cultural_group": "", "topic": "", "context": "", "assertion": "", "source": ""}

Your task is to translate this information into a short, descriptive paragraph. And the detailed explanation of the fields are as follows:
- cultural_group: the country or the cultural group where the cultural knowledge is from
- topic: the topic of the cultural knowledge
- assertion: the most part of the cultural knowledge, which roughly presents the main idea of the cultural knowledge
- context: the supplement of the assertion and provides more detailed information
- source: the source of the cultural knowledge, this is a URL, so you can dive into the URL and check for more information

Notably, you should only use the cultural information above or the URL's information to generate the paragraph, but do not incorporate the URL directly. Your answer should be objective and please do not express any bias or personal opinions.
'''


KNOWLEDGE_PROMPT_USER_TEMPLATE = '''Here is a piece of cultural knowledge encoded in a JSON object:

{}

Based on the information provided, please generate a concise, informative and descriptive paragraph that is easy to understand for others.

Remember, the cultural knowledge above is can only represent a relatively limited scope of the cultural group, so you must not generalize your answer to all the cultural group and your answer should be objective.

Limit your response to 500 words.
'''

KNOWLEGE_EXTRACT_INCONTEXT_DESC = """For the Women problem in Afghanistan, Some Afghan rulers have attempted to increase women's freedom. For the most part, these attempts were unsuccessful. However, there were a few leaders who were able to make some significant changes such as Abdur Rahman Khan. He banned some forms of oppression originating from tribal customs rather than Islam, among them the custom of forcing widows to marry their brother-in-laws, and enforced some rights which Islam did approve of but local tribal customs did not, such as the right of widows to inherit. But his efforts to alleviate women’s oppression caused by tribal customs is limited."""

KNOWLEGE_EXTRACT_INCONTEXT_INCONTEXT_EXP = {
    "cultural group": "Afghanistan",
    "topic": "Women in Afghanistan",
    "assertion": "Some Afghan rulers have attempted to increase women's freedom. For the most part, these attempts were unsuccessful. However, there were a few leaders who were able to make some significant, if temporary, changes.",
    "context": '''Some limited reforms were made by Abdur Rahman Khan, who banned some forms of oppression originating from tribal customs rather than Islam, among them the custom of forcing widows to marry their brother-in-laws, and enforced some rights which Islam did approve of but local tribal customs did not, such as the right of widows to inherit.
                Abdur Rahman Khan, an Afghan ruler, implemented limited reforms in an attempt to alleviate women's oppression caused by tribal customs.''',
    "source": "https://en.wikipedia.org/wiki/Women_in_Afghanistan"
}


QUESTION_GENERATION_SYS_PROMPT = '''You are a helpful expert in generating cultural-aware quetions through cultural knowledge. You are privided with a piece of cultural knowledge and the background of the cultural knowledge. Your task is to generate a single question based on the cultural knowledge that is given to you. The input form is encoded as JSON format, and below is its JSON fields:
{"cultural_group": "", "topic": "", "source": "", "cultural_knowledge": ""}

the detailed explanation of the fields are as follows:
-cultural_group: the country or the cultural group where the cultural knowledge is from
-topic: the topic of the cultural knowledge
-source: the source of the cultural knowledge
-cultural_knowledge: the cultural knowledge that is provided to you, which should pay most attention

----------------------------------------------------
Please strictly follow the following rules:
1. Factuality: Your question should only stems from the cultural knowledge that is provided to you and you shouldn't add other knowledge to your generated question.
2. Specificity: Your question should cover the main idea of the cultural knowledge and should be comprehensive, but not too broad. Try to specific the question with the cultural knowledge and do not ask too general questions.
3. Coverage: You should carefully understand the cultural knowledge and extract the cultural knowledge points as much as possible. And use these cultural knowledge ponints to formulate your question. 
'''

QUESTION_GENERATION_USER_PROMPT = '''Here is a piece of cultural knowledge with its background, your task is to generate a single cultural question based on the cultural knowledge that is provided to you.

----------------------------------------------------
Cultural Knowledge:
{}

Remember, your question should be factually correct and specific to the cultural knowledge. And your question should be comprehensive and cover the main idea of the cultural knowledge. You should only return the question.
'''

ANSWER_GENERATION_SYS_PROMPT = '''You are a helpful consultant for a cultural knowledge question answering scenario. You are given the following question. Your task is to generate a culturally-aware answer to the question.

Remember, your answer should be encoded in JSON format and the detailed explanation of the fields are as follows:
{"answer": "", "cultural_group": "", "language": "", "topic": ""}

- answer: your answer to the question
- cultural_group: the country or the cultural group your answer points to
- language: the language that the cultural group mainly speak
- topic: the main topic of your answer

--------
You should always follow the instructions and directly answer the questions that is provided to you.
'''


ANSER_GENERATION_USER_PROMT = '''You are a helpful consultant for a cultural knowledge question answering scenario. The question is as follows:

{}

You should only return the json object above.

Your Answer:'''

ANSWER_GENERATION_INCONTXT_QUESTION = '''What are the key factors contributing to the restrictions on the freedom and autonomy of women and girls in Afghanistan, and how have initiatives like the Elimination of Violence Against Women (EVAW) law attempted to address the violence they face?'''

ANSWER_GENERATION_IN_CONTEXT_EXAMPLE = '''
{"answer": "In Afghanistan, women and girls face significant restrictions on their freedom and autonomy, largely due to deeply rooted patriarchal customs and traditions. From a young age, they are under the authority of their fathers or husbands, which limits their freedom of movement and choice of spouses. Education and economic opportunities are often denied, further trapping them in cycles of dependence and abuse. Many Afghan women endure domestic violence, with societal stigma and the threat of severe repercussions, including honor killings, deterring them from seeking help or leaving abusive relationships. Efforts to combat such violence led to the creation of the Elimination of Violence Against Women (EVAW) law in 2009, supported by various organizations and activists. However, incidents of extreme violence, such as the public lynching of Farkhunda Malikzada in 2015 and the ongoing attacks on women involved in public life, highlight the persistent dangers they face. Reports from organizations like Amnesty International and Human Rights Watch indicate that violence against women is perpetrated by both state and non-state actors, and women with disabilities encounter additional discrimination. The situation remains grave, as exemplified by the assassination attempt on Fawzia Koofi, a member of Afghanistan's peace negotiation team, reflecting the continued threats against women's rights advocates in the country.",
"cultural_group": "Afghanistan Women",
"language":"Pashto, Dari",
"topic": "Women in Afghanistan"}
'''

TRANSLATE_PROMPT_SYS = '''You are a helpful translator for a cultural knowledge question answering system. You are given a question and answer in English and You should translate the question and answer into the language of {}.

The input text will be encoded in JSON format, and you should also return the translated text in JSON format. You should not translate the JSON fields, only the text should be translated.
And you should not translate the text which is in the JSON fields of "cultural_group", "topic", "source", "question_idx", you should not to forget to translate "cultural_knowledge".


Your translated text should also be in the JSON format.

Here is JSON format that you should translate:
{}
'''



ANSWER_GENERATION_SYS_PROMPT_GPT = '''You are a helpful consultant for a cultural knowledge question answering scenario. You are given the following question and its cultural knowledge. Your task is to generate a culturally-aware answer to the question based on the cultural knowledge.

Remember, your answer should be encoded in JSON format and the detailed explanation of the fields are as follows:
{"answer": "", "cultural_group": "", "language": "", "topic": ""}

- answer: your answer to the question
- cultural_group: the country or the cultural group your answer points to`
- language: the language that the cultural group mainly speak
- topic: the main topic of your answer

--------
Notably, the question stems from the cultural knowledge, so your answer should also stem from the cultural knowledge provided to yuo. You should always follow the instructions and directly answer the questions that is provided to you.
'''


ANSWER_GENERATION_USER_PROMPT_GPT = '''You are a helpful consultant for a cultural knowledge question answering scenario. The question is as follows:

{}

the cultural knowledge is as follows:

{}

Remember, your answer should correlate with the cultural knowledge . You should only return the answer.

Your Answer:
'''

ANSWER_GENERATION_INCONTEXT_KNOWLEDGE = '''In Afghanistan, women and girls face significant restrictions on their freedom and autonomy, largely due to deeply rooted patriarchal customs and traditions. From a young age, they are under the authority of their fathers or husbands, which limits their freedom of movement and choice of spouses. Education and economic opportunities are often denied, further trapping them in cycles of dependence and abuse. Many Afghan women endure domestic violence, with societal stigma and the threat of severe repercussions, including honor killings, deterring them from seeking help or leaving abusive relationships. Efforts to combat such violence led to the creation of the Elimination of Violence Against Women (EVAW) law in 2009, supported by various organizations and activists. However, incidents of extreme violence, such as the public lynching of Farkhunda Malikzada in 2015 and the ongoing attacks on women involved in public life, highlight the persistent dangers they face. Reports from organizations like Amnesty International and Human Rights Watch indicate that violence against women is perpetrated by both state and non-state actors, and women with disabilities encounter additional discrimination. The situation remains grave, as exemplified by the assassination attempt on Fawzia Koofi, a member of Afghanistan's peace negotiation team, reflecting the continued threats against women's rights advocates in the country.'''


ANSWER_EXTRACT_PROMPT_SYS = ''''You are a helpful assistant for a cultural knowledge question answering scenario. You are provided with a cultural question and an anwer to that question. Your task is to  extract the meaningful cultural knowledge points from the answer\
A Knowledge point is  an atomic statement that only contains a fine-grained information unit of the answer and you should generate all the meaningful knowledge points from the answer.
Notably, each knowledge point should be a single sentence and should be concise and self-contained.

The input form is encoded as JSON format, and below is its JSON fields:
{"question": "", "answer": ""}
- question: the cultural question given to you
- answer: the answer to the cultural question

Your return should also be encoded in JSON format, with each knowledge point as a list of strings,which will be like:
{"knowledge_points": ["knowledge_point1", "knowledge_point2", ...]}

And here are an example for your extraction task:
<example start/>
input:
{"question": "How did the establishment and expansion of the Mughal Empire under Babur influence the religious dynamics and architectural developments in South Asia during the 16th century?"
"answer": "The establishment and expansion of the Mughal Empire under Babur significantly influenced the religious dynamics and architectural developments in South Asia during the 16th century. Founded by Babur, a ruler of Turco-Mongol descent, the empire primarily adhered to Sunni Islam and expanded its territory across the northwest and Indo-Gangetic Plain regions. While much of the southern and northeastern regions were under Hindu kings, the Mughals defeated Rajput kingdoms and the Vijayanagara Empire, leading to a period marked by architectural achievements such as the iconic Taj Mahal. However, this period also experienced significant religious persecution, with many Buddhist, Hindu, and Sikh temples facing desecration, and the imposition of the jizya, a religious tax on non-Muslims, highlighting the challenges faced by these communities. The executions of Sikh leaders Guru Arjan and Guru Tegh Bahadur after their refusal to convert to Islam further exemplified the tense religious dynamics of the time."
}
Your extracted knowledge points:
{"knowledge_points": ["The Mughal Empire was founded by Babur, a ruler of Turco-Mongol descent.",
"The Mughal Empire primarily adhered to Sunni Islam.",
"The Mughal Empire expanded its territory across the northwest and Indo-Gangetic Plain regions.",
"Much of the southern and northeastern regions of India were under Hindu kings during the 16th century.",
"The Mughals defeated Rajput kingdoms and the Vijayanagara Empire.",
"The establishment of the Mughal Empire led to significant architectural achievements, including the construction of the Taj Mahal.",
"The period of Mughal rule was marked by religious persecution of various communities.",
"Many Buddhist, Hindu, and Sikh temples faced desecration during Mughal rule.",
"The imposition of the jizya, a religious tax on non-Muslims, occurred during the Mughal Empire.",
"The executions of Sikh leaders Guru Arjan and Guru Tegh Bahadur were significant events during this period.",
"The executions of these Sikh leaders occurred after their refusal to convert to Islam, highlighting religious tensions."
]}

Please Strictly follow the instructions and generate your own extracted knowledge points.
Last but not least, you should extract the knowledge points in the language of the answer's language, which is pretty important.
'''

ANSWER_EXTRACT_PROMPT_USER = '''input:
{}

Your extracted knowledge points:
'''


CRITIQUE_PROMPT_SYS = '''You are an expert reviewer for a cultural knowledge question answering system. You have plenty of cultural knowledge in {}.

You are given a JSON object and the detailed explanation of the fields are as follows:
{"question": "", "grounded_answer": "", "answer_to_critique": "", "grounded_answer_knowledge_points": "", “knowledge_points_to_critique”: ""}
-question: the cultural question that is given to you
-grounded_answer: the grounded answer to the question, which is the reference answer
-answer_to_critique: the answer that you should critique
-grounded_answer_knowledge_points: the knowledge points extracted from the grounded answer, each knowledge point is a single sentence and is seperated with a comma in a list
-knowledge_points_to_critique: the knowledge points extracted from the answer_to_critique, each knowledge point is a single sentence and is seperated with a comma in a list

Your task is to give a detailed critique based on the answer_knowledge_points and the grounded_answer_knowledge_points point by point. Your detailed critique should have the ability to point out the flaws of the answer_to_critique and provide constructive suggestions for improvement.


Notably, you should compare the grounded_answer_knowledge_points  and the answer_knowledge_points and provide a detailed critique based on the comparison. And your critique should based on the principles below:
1. Correctness: Be sure to point out any factual inaccuracies or errors in the answer_to_critique and provide corrections based on the grounded_answer_knowledge_points.
2. Comprehensiveness: The answer_to_critique should cover the main points of the grounded_answer and should not miss any key information, if the answer_to_critique miss the cultural knowledge points, you should say "not addressed clearly" between the comparison.
3. Stability: If the grounded_answer_knowledge_points and the knowledge_points_to_critique are mainly the same, you should say "Roughly the same" in your critique.
4. Point by point: You should compare the grounded_answer_knowledge_points and knowledge_points_to_critique point by point and provide your critique based on the comparison. Between the comparison, you should choose the most relevant knowledge_points_to_critique from the list while comparing the grounded_answer_knowledge_points.

You should always follow the instructions and carefully compare the grounded_answer_knowledge_points and the answer_knowledge_points point by point and provide your critique.
Remember, you should directly compare the grounded_knowledge_points and knowledge_points_to_critique and point out the flaws made by knowledge_points_to_critique.

And your critique should be encoded in JSON format, with each unit be a small JSON format , which will be like:
{
    {
        "grounded_answer_knowledge_points": "",
        "knowledge_points_to_critique": “ ”
        "Critique": ""
    },
    {
        "grounded_answer_knowledge_points": "",
        "knowledge_points_to_critique": “ ”
        "Critique": ""
    },
    .....
}


Here are an example for your critique task:
<example start/>
input:
 { “question”: ”What role do annual fairs at shrines play in the lives of Afghan pilgrims, particularly women, despite the traditional Islamic discouragement of practices such as saint veneration and shrine visitation?”, 
 “grounded_answer”: “In Afghanistan, annual fairs at shrines play a significant role in the lives of Afghan pilgrims, especially women. These festive gatherings attract thousands of individuals who seek the intercession of saints for special favors, such as healing from illness or the desire for a child. Women demonstrate a strong devotion to the activities associated with these shrines, often participating actively in the fairs. Pilgrims might visit for a short time or extend their stay for several days, and they often leave with specially blessed amulets, known as tawiz, which are believed to provide protection from the evil eye and foster loving relationships. Despite the traditional Islamic discouragement of practices like saint veneration and shrine visitation, these fairs continue to hold cultural significance for many, offering a sense of community and spiritual fulfillment”, 
 “answer_to_critique”: “In Afghanistan, annual fairs at shrines are an integral part of the country's rich cultural heritage, serving as significant events for both men and women. These fairs, often taking place around sacred sites such as the shrine of Khwaja Abdullah Ansari in Herat, provide pilgrims with opportunities to connect with their faith, community, and cultural traditions. For Afghan women, these events can be particularly empowering, as they offer a chance to engage in social interactions, seek blessings and healing, and participate in festive activities that transcend traditional Islamic discouragement of shrine visitation. Women often come to these fairs to pray, seek guidance from spiritual leaders, and participate in traditional practices such as the distribution of bread and sweets. The fairs also serve as a platform for women to network, share stories, and access information on healthcare, education, and other essential services. The spiritual and social aspects of these events help to balance the more conservative aspects of Afghan society, providing women with a sense of belonging and agency within their community.” 
“grounded_answer_knowledge_points”:  “["Annual fairs at shrines are significant for Afghan pilgrims, especially women.",
"These fairs attract thousands of individuals seeking the intercession of saints for special favors.",
"Common requests at these fairs include healing from illness and the desire for a child.",
"Women actively participate in the activities associated with shrine fairs.",
"Pilgrims may visit for a short time or extend their stay for several days at the fairs.",
"Pilgrims often leave the fairs with specially blessed amulets known as tawiz.",
"Tawiz are believed to provide protection from the evil eye and foster loving relationships.",
"Despite traditional Islamic discouragement of saint veneration and shrine visitation, these fairs hold cultural significance.",
"The fairs offer a sense of community and spiritual fulfillment for many attendees.",
"Annual fairs are part of the cultural practices of Afghan pilgrims who speak Pashto and Dari."]”,
 “knowledge_points_to_critique”: “["Annual fairs at shrines are an integral part of Afghanistan's cultural heritage.", 
 "These fairs serve as significant events for both men and women in Afghanistan.",
  "Fairs often take place around sacred sites, such as the shrine of Khwaja Abdullah Ansari in Herat.", 
  "Pilgrims connect with their faith, community, and cultural traditions during these fairs.", 
  "For Afghan women, shrine fairs can be particularly empowering.", 
  "Women engage in social interactions at these fairs, seeking blessings and healing.", 
  "Women participate in festive activities that challenge traditional Islamic discouragement of shrine visitation.", 
  "At these fairs, women pray and seek guidance from spiritual leaders.", 
  "Women participate in traditional practices, such as the distribution of bread and sweets, at the fairs.",
   "The fairs provide a platform for women to network and share stories.", 
   "Women can access information on healthcare, education, and essential services at these events.", 
   "The spiritual and social aspects of the fairs help balance the conservative aspects of Afghan society.", 
   "These events provide women with a sense of belonging and agency within their community."]" } 

Your critique:
</example start>
{
    {
        "grounded_answer_knowledge_points": "Annual fairs at shrines are significant for Afghan pilgrims, especially women.",
        "knowledge_points_to_critique": "Annual fairs at shrines are an integral part of Afghanistan's cultural heritage.",
        "Critique": "While both points acknowledge the importance of the fairs, the grounded answer emphasizes the specific significance for women, which is a crucial aspect of the question. The answer_to_critique could improve by explicitly mentioning how these fairs are particularly significant for women, thereby enhancing its focus on the gender aspect. Attracting Thousands of Individuals"
    },
    {
        "grounded_answer_knowledge_points": "These fairs attract thousands of individuals seeking the intercession of saints for special favors.",
        "knowledge_points_to_critique": These fairs serve as significant events for both men and women in Afghanistan.”
        "Critique": "Roughly the same"
    },
    {
        "grounded_answer_knowledge_points": “Common requests at these fairs include healing from illness and the desire for a child.”,
        "knowledge_points_to_critique": "Not addressed directly."
        "Critique": "This aspect is entirely missing in the answer_to_critique. Including this detail would enhance comprehensiveness by highlighting the specific motivations behind why individuals, especially women, attend these fairs, thereby enriching the cultural context."
    },
    {
        "grounded_answer_knowledge_points": "Women actively participate in the activities associated with shrine fairs.",
        "knowledge_points_to_critique": “For Afghan women, shrine fairs can be particularly empowering.”,
        "Critique": "Roughly the same"
    },
    {
        "grounded_answer_knowledge_points": "Pilgrims may visit for a short time or extend their stay for several days at the fairs.",
        "knowledge_points_to_critique": "Not addressed directly."
        "Critique": "This point is absent in the answer_to_critique, which detracts from its comprehensiveness. Including information about the duration of visits would illustrate the fairs’ importance as a social and spiritual gathering, emphasizing the depth of engagement among attendees."
    },
    {
        "grounded_answer_knowledge_points": "Tawiz are believed to provide protection from the evil eye and foster loving relationships.",
        "knowledge_points_to_critique": "Not addressed directly."
        "Critique": "The answer_to_critique omits any mention of the beliefs associated with tawiz, which is vital to understanding the spiritual significance of the fairs. Including this aspect would contribute to a more culturally aware response."
    },
    {
        "grounded_answer_knowledge_points": "Despite traditional Islamic discouragement of saint veneration and shrine visitation, these fairs hold cultural significance.",
        "knowledge_points_to_critique": “Women participate in festive activities that challenge traditional Islamic discouragement of shrine visitation.”,
        "Critique": "The answer_to_critique does address the challenge to traditional discouragement but fails to contextualize it within the broader cultural significance of the fairs as highlighted in the grounded answer. It would be beneficial to acknowledge that many continue to find meaning in these practices despite discouragement, thus providing a more nuanced view."
    },
    {
        "grounded_answer_knowledge_points": "The fairs offer a sense of community and spiritual fulfillment for many attendees.",
        "knowledge_points_to_critique": “The spiritual and social aspects of the fairs help balance the conservative aspects of Afghan society.”
        "Critique": "While the answer_to_critique discusses the balancing effect on conservative aspects, it misses the explicit mention of community and spiritual fulfillment, which are central to the grounded answer. A more comprehensive discussion of how these fairs fulfill both communal and spiritual needs would enhance the answer’s depth. Cultural Practices of Afghan Pilgrims"
    },
    {
        "grounded_answer_knowledge_points": "Annual fairs are part of the cultural practices of Afghan pilgrims who speak Pashto and Dari.",
        "Answer Knowledge Point": "Not addressed directly.",
        "Critique": "This point about the cultural practices tied to language is entirely overlooked. Including it would add a layer of cultural awareness and show the linguistic diversity present at these events."
    }
}
<example end/>
'''

CRITIQUE_PROMPT_USER = '''
input:
{}

Your critique:

'''

CRITIQUE_SUMMARIZATION_PROMPT_SYS = '''You are an expert summarizer for a cultural knowledge question answering critique. Your are given a point-by-point critique of a cultural knowledge question answering critique. Your task is to summarize the critique and provide a concise and informative summary of the critique.

The critique are mainly classified into three class: Not addressed directly, Roughly the same, and the detailed critique. 
-For the "Not addressed directly" class, your critique should point out the missing cultural knowledge point.
-For the "Roughly the same" class, you can choose to not provide any critique.
-For the detailed critique, your critique should provide some detailed critique based on the comparison of the grounded_answer_knowledge_points and the knowledge_points_to_critique.
You just know the three classes and should not explicit the three classes' name in your answer. And you should summarize the critique as if you are giving the critique to the answer. You should provide a concise and informative summary of the critique.

the input is encoded in JSON format, and the detailed explanation of the fields are as follows,which is the point-by-point critique of the cultural knowledge question answering critique:
{
    {
        "grounded_answer_knowledge_points": "",
        "knowledge_points_to_critique": “ ”
        "Critique": ""
    },
    {
        "grounded_answer_knowledge_points": "",
        "knowledge_points_to_critique": “ ”
        "Critique": ""
    },
    .....
}

You should directly give the specific critique based on the comparison and should not start with the general statement.

Remember, your summary should be informative and useful for refinement as if you are directly critiquing the answer. Your should only return the critique you summarized.
'''

CRITIQUE_SUMMARIZATION_PROMPT_USER = '''
input:
{}

Your summary:
'''

EVAL_CULTURAL_POINTS_PROMPT = '''You are an expert evaluator for a cultural knowledge question answering system. You are given a piece of cultural knowledge point and a list of reference cultural knowledge. Your task is to evaluate whether the given cultural knowledge point satisfies one of the reference cultural knowledge points and give a concise explanation.

Here are some examples and explanations:
</example1 start>
cultural knowledge points:
"The centers aim to improve literacy rates among Afghan citizens, particularly women and children."

reference cultural knowledge points:
["Lincoln learning centers in Afghanistan improve literacy rates among Afghan citizens.", 
"These centers were established in response to low literacy rates in Afghanistan.", 
"Lincoln learning centers serve as educational hubs providing English language classes, library facilities, Internet connectivity, and counseling services.", 
"The initiative aims to reach at least 4,000 Afghan citizens each month at each location.", 
"Literacy courses are mandatory for the military and national police forces in Afghanistan.",
 "The initiative reflects a broader commitment to enhancing literacy levels across Afghanistan.", 
 "Educational programs at the centers promote an understanding of American culture.", 
 "The primary languages spoken in the Lincoln learning centers are Dari and Pashto."]

Your output:
Yes

explanation: the cultural knowledge point "The centers aim to improve literacy rates among Afghan citizens, particularly women and children." is similar to the reference cultural knowledge point "These centers were established in response to low literacy rates in Afghanistan.", so the output is Yes.
</example1 end>

</example2 start>
cultural knowledge points:
"Basketball is gaining popularity in Afghanistan and is enjoyed by both men and women."

reference cultural knowledge points:
["The Afghan Sports Federation was established in 1922.",
"The Afghan Sports Federation promotes sports like football and basketball in Afghanistan.",
"The federation is responsible for developing, organizing, and overseeing various sports in Afghanistan.",
"Afghanistan's national football team qualified for the 2014 FIFA World Cup.",
"The qualification for the 2014 FIFA World Cup was a significant milestone for Afghan football.",
"The Afghan Sports Federation faces challenges such as financial constraints and infrastructure limitations.",
"Ongoing conflicts in Afghanistan have impacted the development of sports.",
"The history of the national football team reflects Afghanistan's turbulent past.",
"Many players and coaches of the national football team have fled Afghanistan due to conflict or persecution.",
"The Taliban banned sports during their rule from 1996 to 2001.",
"The ban on sports during Taliban rule hindered the progress of the national football team.",
"The national football team has shown resilience despite numerous challenges.",
"Players like Zohib Islam Amiri and Faisal Hamidi have represented Afghanistan in international competitions.",
"The 2021 Taliban takeover has raised concerns about the future of sports in Afghanistan."]

Your output:
No

explanation: the cultural knowledge point "Basketball is gaining popularity in Afghanistan and is enjoyed by both men and women" is not addressed clear in the reference cultural knowledge points, so the output is No.
<example2 end/>

Remember, Your output should first generate 'Yes' or 'No', and give a concise explanation of your evaluation. 
If your answer is "Yes", your explanation should specifically incorporate the given cultural knowledge point satisfies which reference cultural knowledge point.

cultural knowledge points:
{}

reference cultural knowledge points:
{}

Your output:
'''

EVAL_TOPIC__PROMPT = '''You are an expert evaluator for a cultural knowledge question answering system. You are given two topics and you should judge whether the two topic are roughly the same.
Here is an example:
</example start>
topics:
"Culture of Afghanistan"
"Afghanistan's Cultural Diversity and Heritage"

your output:
Yes
</example end>

Remember, Your output should always be "Yes" or "No", and should not contain other words.

topics:
{}
{}

Your output:
'''

EVAL_TOPIC_PROMPT = '''You are an expert evaluator for a cultural knowledge question answering system. You are given two topics and you should judge whether the two topic are roughly the same.

Here is an example:
</example start>
topics:
"Culture of Afghanistan"
"Afghanistan's Cultural Diversity and Heritage"

your output:
Yes
<example end/>

Remember, Your output should always be "Yes" or "No", and should not contain other words.

topics:
{}
{}

Your output:
'''

EVAL_CULTURAL_GROUP_PROMPT = '''You are an expert evaluator for a cultural knowledge question answering system. You are given two cultural groups and you should judge whether the two cultural groups are roughly the same.
Here is an example:

</example start>
cultural groups:
"Afghan Proverbs and Communication"
"Afghan proverbs"

your output:
Yes
<example end/>

Remember, Your output should always be "Yes" or "No", and should not contain other words.

cultural_group:
{}
{}

Your output:
'''

EVAL_CULTURAL_LANGUAGE_PROMPT = '''You are given a language and a list. Your task is to judge whether the language corresponds any item in the list. If the language corresponds any item in the list, you should return "Yes", otherwise, you should return "No".
langauge: {}
language list: {}
Remember, Your output should always be "Yes" or "No", and should not contain other words.
Your output:
'''

ERROR_FIX_PROMPT = '''You are an expert error fixer. And You need to fix this error about the JSON format.
You need to read the error and change the object to JSON format to make it correct.

Error
{}

Object
{}

You should only return the correct JSON format of the object with the filed name and value.

'''

JSON_REFORMAT_PROMT = '''You are an expert JSON reformatter. You are given a string and you should parse the string to JSON format.
the fields are 
'''

ANSWER_LANGUAGE_GENERATION_PROMPT = '''You are a helpful consultant for a cultural knowledge question answering scenario. You are given the following question. Your task is to generate the cultural group of the question, the language that the cultural group mainly speak based on the question and the topic of the quesiton.
Remember, your answer should be encoded in JSON format and the detailed explanation of the fields are as follows:
/{"cultural_group": "", "language": "", "topic": ""}/

- cultural_group: the country or the cultural group your answer points to
- language: the language that the cultural group mainly speak
- topic: the main topic of your answer

Here is an example for your task:
<example>
Question: ''What are the key factors contributing to the restrictions on the freedom and autonomy of women and girls in Afghanistan, and how have initiatives like the Elimination of Violence Against Women (EVAW) law attempted to address the violence they face?''

Your answer : {"cultural_group": "Afghanistan Women","language":"Pashto, Dari", "topic": "Women in Afghanistan"}
</example>
'''

ANSWER_LANGUAGE_GENERATION_PROMPT_USER = '''Question: {}
Your answer:
'''


KNOWLEDGE_ENTAIL_SYSTEM_PROMPT = """You are an AI assistant trained to classify whether a given response from a language agent entails a specific cultural knowledge.

Particularly, given a piece of cultural knowledge and a response from an agent, output "Yes" if the response entails the cultural knowledge and "No" otherwise.

Your output should be Yes/No only. Even if you are uncertain, you must pick either "Yes" or "No" without using any other words."""

KNOWLEDGE_ENTAIL_USER_TEMPLATE = """Response: {}

Knowledge: {}

Does the given response entail the provided knowledge?

Entailment (Yes/No):"""


TRANSLATE_EVALUATE_USR_PROMPT = '''You are a helpful evaluator for a translate task. You are given a translated text and the original text in English. Your task is to evaluate whether the translated text is correct or not.
The translated text should be fluent and accurate, and should convey roughly the same meaning as the original text. If the translated text is fluent and accurate, you should only return "Yes", otherwise, you should return "No", and provide some suggestions or critique to help increase the quality of the translation.
original text:
{}
translated text:{}
Your answer:
'''


TRANSLATE_PROMPT_USER = '''You are a helpful translator for a cultural knowledge question answering system. You are given a text in English and You should translate the text into the language of {}.

Here is text that you should translate:
{}
'''

REFINE_PROMPT_USER = '''You are a helpful refiner for a cultural knowledge question answering system. You are given a question and answer pair, and a critique summary about the answer. Your task is to refine the answer based on the critique provided.

Here is an example:
<example start/>
question: What role does the Afghan Sports Federation play in promoting sports like football and basketball in Afghanistan, and how has the history of the national football team reflected the country's ongoing conflicts?
answer: The Afghan Sports Federation plays a crucial role in promoting sports like football and basketball in Afghanistan. Established in 1922, it is responsible for developing, organizing, and overseeing various sports across the country. The federation has contributed to the growth of football, with Afghanistan's national team qualifying for the 2014 FIFA World Cup, marking a significant milestone. Despite facing numerous challenges, including financial constraints, infrastructure limitations, and ongoing conflicts, the federation has managed to promote sports development in Afghanistan. However, the history of the national football team reflects the country's turbulent past, with many players and coaches fleeing the country due to conflict or persecution. The team's progress has been hindered by the Taliban's ban on sports during their rule from 1996 to 2001 and the subsequent lack of infrastructure and resources. Despite these challenges, the national team has shown resilience, with players like Zohib Islam Amiri and Faisal Hamidi representing the country in international competitions. The federation's efforts to promote sports have been further hindered by the 2021 Taliban takeover, which has raised concerns about the future of sports in Afghanistan.
critique summary: The critique highlights several key areas where the answer could be improved for greater cultural knowledge accuracy and depth. \n\n1. The popularity of football in Afghanistan is a significant omission that should be included to provide context about its cultural importance.\n2. While the establishment dates of the Afghan national football team and the Afghan Sports Federation are mentioned, the answer should clarify that the 1922 reference specifically pertains to the national football team to better understand the sports context.\n3. The historical fact of Afghanistan's membership in FIFA in 1948 is missing, which would enhance the comprehensiveness of the answer regarding the country's football history.\n4. The critique points out that while the impact of ongoing conflicts on sports development is noted, the specific gap in international competition for the national football team from 1984 to 2003 should be addressed for clarity on the challenges faced.\n5. The formation of the women's national football team in 2007 is another critical point that is missing, highlighting progress in women's sports in Afghanistan.\n6. The significance of Ghazi Stadium, both historically and in its current use for football, is not mentioned, which would provide valuable insight into Afghanistan's sports history.\n7. The resilience of sports in Afghanistan, as showcased through the use of Ghazi Stadium, is an important theme that the answer fails to reflect.\n8. Several points regarding basketball, its historical presence since the 1970s, its growing popularity, and its role in cultural acceptance are not addressed, all of which are crucial for a comprehensive view of sports culture in Afghanistan.\n\nOverall, including these missing points and clarifications would significantly enrich the understanding of sports' cultural significance in Afghanistan.
Your refined answer: The Afghan Sports Federation plays a crucial role in promoting various sports, including football and basketball, across Afghanistan. It organizes activities and events to foster athletic development and participation. Football, being the most popular sport in the country, has a significant history; the national football team was established in 1922 and became a member of FIFA in 1948. However, due to ongoing conflicts, the team did not compete internationally from 1984 until 2003. This history reflects the challenges faced by the country, as the national team works to re-establish itself globally. The women's national football team was formed in 2007, symbolizing progress despite the difficult context. Additionally, the Ghazi Stadium, which has a controversial past linked to the Taliban's rule, is currently a venue for football matches, showcasing the resilience of sports in the face of adversity. Basketball, which has been played since at least the 1970s, is also gaining popularity and is enjoyed by both men and women, contributing to a broader cultural acceptance of sports in Afghanistan
<example end/>


Here is the question and answer pair and the critique:
question: {}
answer: {}
critique: {}

Your refined answer:
'''