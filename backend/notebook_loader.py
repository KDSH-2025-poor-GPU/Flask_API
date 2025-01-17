import google.generativeai as genai
import os
import time
import re

def generate_optimized_prompt(model, user_prompt):
    response=model.generate_content(user_prompt)
    time.sleep(5)
    return response.text

def score_paper(paper_text, sub_task, model, max_retries=5, delay=5):

    prompt = f"""
    Sub-task: {sub_task}

    Evaluate the following research papers for content text for its {sub_task} on a scale of 1 to 10 (You can include fractional too).
    Just give scores. No need for explanation.

    Paper Text:
      {paper_text}

    Please provide a score on a scale from 1 to 10, with no further explanation or text.
    Just the score:
    """

    attempt = 0
    while attempt < max_retries:
        try:

            response = generate_optimized_prompt(model, prompt)


            if response is None:
                print(f"Attempt {attempt + 1}: No response received for subtask: {sub_task}. Retrying...")
                attempt += 1
                time.sleep(delay)
                continue


            if "error" in response:
                print(f"Attempt {attempt + 1}: Error in API response: {response['error']}. Retrying...")
                attempt += 1
                time.sleep(delay)
                continue


            match = re.search(r"(\d+(\.\d+)?)", response)
            if match:
                score = float(match.group(1))
                return score


            print(f"Attempt {attempt + 1}: Could not extract a score from response: {response}. Retrying...")
            attempt += 1
            time.sleep(delay)
        except Exception as e:
            print(f"Attempt {attempt + 1}: Exception occurred: {e}. Retrying...")
            attempt += 1
            time.sleep(delay)


    print(f"All {max_retries} attempts failed for subtask: {sub_task}. Returning -1.")
    return -1
def generate_scores(paper_text, model):
  scores={}
  criteria_dict={
      "Originality" : """"Criteria :
                          - How novel is the approach presented in this paper?
                          - Does the paper introduce new methods or significantly improve existing ones?
                          - Can you identify any gaps or limitations in the related work section?""" ,
      "Methodology" : """Criteria :
                          - How well-structured and transparent are the experimental design, data collection methods, and analysis procedures?
                          - Are the results accurately interpreted and supported by the data?
                          - Can you identify any potential biases or methodological flaws?""",
      "Clarity and Concision" : """Criteria :
                                    - How clear and concise is the writing style throughout the paper?
                                    - Are the figures, tables, and illustrations effectively used to communicate complex information?
                                    - Can you easily follow the logical flow of the arguments presented in the paper?""",
      "Impact" : """Criteria :
                      - How significant are the implications of the findings for practice, policy, or future research?
                      - Does the paper address a pressing problem or gap in the field?
                      - Can you identify any potential applications or extensions of the research?""",
      "Novel application or extension" : """Criteria :
                      - How effectively does the paper apply existing methods to a new domain, problem, or dataset?
                      - Does the paper extend current techniques in meaningful ways?
                      - Can you identify any potential limitations or areas for future work?
      """

      }
  for sub_task,criteria in criteria_dict.items():
    score_response = score_paper(paper_text ,sub_task, model)
    scores[sub_task]=score_response

  print(scores)
  return scores