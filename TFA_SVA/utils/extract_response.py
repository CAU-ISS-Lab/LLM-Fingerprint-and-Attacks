import re
def gsm_extract_math_answer(pred_str):
    try:
        if( 'boxed' in pred_str):
            ans = pred_str.split('boxed')[-1]
        elif('the answer is ' in pred_str):
            ans = pred_str.split('the answer is ')[-1].strip()
        elif 'The answer is ' in pred_str:
            ans = pred_str.split('The answer is ')[-1].strip()
        else:
            ans = pred_str

        pattern = r'-?\d*[\.,]?\d+'
        pred = re.findall(pattern, ans)

        if(len(pred) >= 1):
            # print(pred_str)
            pred = float(pred[-1].replace(',',''))
        else:
            pred = float("nan")


    except Exception:
        print(f"Cannot parse the resulting num in predicted solution {pred_str}.\n")
        pred = float("nan")

    return pred
