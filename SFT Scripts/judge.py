import os
from openai import OpenAI
import pandas as pd
import tqdm
import re

client = OpenAI(api_key="")

df = pd.read_csv("")

system_prompt = """
Please act as an impartial judge and evaluate whether the LLM's response is factually correct based solely on the provided atomic fact and that it answers the question.

Evaluation Scale:
• Class 1: CORRECT - The LLM's response contains information that is consistent with the atomic fact provided and answers what asked in the question.
• Class -1: INCORRECT - The LLM's response contradicts or misrepresents the atomic fact provided. Even if factually correct, if the response does not answer the question as asked, classify as INCORRECT.

Your evaluation must be based EXCLUSIVELY on the atomic fact. Do not use any external knowledge or information beyond what is explicitly stated in the atomic fact. The atomic fact is the only source of truth for this evaluation.

IMPORTANT:
• A response should be classified as CORRECT if the response faithfully reflects the atomic fact and directly addresses the question.
• Only classify a response as INCORRECT if it directly contradicts the atomic fact or presents information that is inconsistent with the atomic fact or does not answer what's asked in the question.
• Extra details beyond the atomic fact should not affect correctness unless they introduce contradictions.
• If the response addresses the question with information that aligns with the atomic fact, classify it as CORRECT regardless of any supplementary details. Do not add any information from your end. Only answer based on the provided evaluation criteria. Do
not check for anything extra like completeness or style.

Answer Format:
• Class 1 (CORRECT): <reasoning for why the LLM's response accurately reflects the information in the atomic fact and provides a correct response to the user>
• Class -1 (INCORRECT): <reasoning for why the LLM's response contradicts or misrepresents the atomic fact>


Final Verdict: <assigned class> (1/-1)
Explanation: Based on the atomic fact provided, explain why the response is assigned to the final class in 2-3 lines.

"""

def extract_class(text):
    lower_text = text.lower()
    target = "final verdict"

    start_index = lower_text.find(target)
    if start_index == -1:
        return None  

    start_index += len(target)
    end_index = text.find('\n', start_index)
    if end_index == -1:
        end_index = len(text) 

    substring = text[start_index:end_index]
    match = re.search(r'-?\s*\d+', substring)
    if match:
        assigned_class = int(match.group(0).replace(' ', ''))
        return assigned_class
    else:
        return None 
    
def format_prompt(question, llm_response , atomic_fact):
    prompt = f"""Atomic Fact: {atomic_fact}

    Question: {question}

    Response:{llm_response}
    """
    return prompt  



list_df = ["your_path/file1.csv",
           "your_path/file2.csv"
]

for file_name in list_df:

    title = file_name.split("/")[-1].replace(".csv", "")
    df = pd.read_csv(file_name)
    answer_rphr_3_judged = []
    answer_rphr_6_judged = []


    for i in tqdm.tqdm(range(len(df))):
        for j in range(2):
            if j == 0:
                question = df.loc[i, "rephrased_3"]
                llm_response = df.loc[i, "answer_rphr_3"]
                atomic_fact = df.loc[i, "fact"]
            else:
                question = df.loc[i, "rephrased_6"]
                llm_response = df.loc[i, "answer_rphr_6"]
                atomic_fact = df.loc[i, "fact"]

            prompt = format_prompt(question,llm_response , atomic_fact)
            response = client.responses.create(
                model="gpt-4o-mini",
                instructions= system_prompt,
                input=prompt,
                max_output_tokens=500,
                temperature=0.1)

            response = response.output_text.strip()
            answer = extract_class(response)
            if j == 0:
                answer_rphr_3_judged.append(answer)
            else:
                answer_rphr_6_judged.append(answer)


    df["answer_rphr_3_judged"] = answer_rphr_3_judged
    df["answer_rphr_6_judged"] = answer_rphr_6_judged

    correct_3 = sum(df["answer_rphr_3_judged"] == 1)
    correct_6 = sum(df["answer_rphr_6_judged"] == 1)
    total = len(df)
    both_correct = sum((df["answer_rphr_3_judged"] == 1) & (df["answer_rphr_6_judged"] == 1))
    one_wrong = sum(((df["answer_rphr_3_judged"] == 1) & (df["answer_rphr_6_judged"] == -1)) |
                    ((df["answer_rphr_3_judged"] == -1) & (df["answer_rphr_6_judged"] == 1)))
    both_wrong = sum((df["answer_rphr_3_judged"] == -1) & (df["answer_rphr_6_judged"] == -1))

    print(f"\n\nSummary for {title}:")
    print(f"  Rephrase 1 correct: {correct_3 / total * 100:.2f}%")
    print(f"  Rephrase 2 correct: {correct_6 / total * 100:.2f}%")
    print(f"  Both correct: {both_correct / total * 100:.2f}%")
    print(f"  One wrong: {one_wrong / total * 100:.2f}%")
    print(f"  Both wrong: {both_wrong / total * 100:.2f}%\n")


    df.to_csv(f"/Reward_Gen/sft/results/{title}_Judged_Rerun.csv", index=False)