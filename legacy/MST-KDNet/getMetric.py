import re

def extract_scores(input_file, output_file):
    try:
        # 正则表达式匹配 "Epoch 1 validation (hd95|dice) score for modality <modality> >> <score>, core<core_score>, enhance<enhance_score>"
        # 'Epoch 1 validation hd95 score for  modality t1t2>> 7.351554690947929 ,core0.31169891973005615,enhance0.2727215913605943'
        pattern = r"Epoch 1 validation (hd95|dice) score for  modality ([\w\d]+)>> ([\d\.]+) ,core([\d\.]+),enhance([\d\.]+)"
        
        # 打开输入文件读取
        with open(input_file, 'r') as infile:
            lines = infile.readlines()
        
        # 用正则表达式提取所有符合条件的行
        extracted_data = []
        for line in lines:
            line = line.strip()  # 去掉首尾的空白字符（包括换行符）
            match = re.search(pattern, line)
            if match:
                score_type = match.group(1)  # 指标类型 ('hd95' 或 'dice')
                modality = match.group(2)  # 模态组合
                score = match.group(3)  # 分数
                core_score = match.group(4)  # core分数
                enhance_score = match.group(5)  # enhance分数
                
                # 格式化输出内容
                extracted_data.append(f"score type: {score_type}, modality: {modality}, score: {score}, core score: {core_score}, enhance score: {enhance_score}\n")
        
        # 将提取的数据写入到输出文件
        with open(output_file, 'w') as outfile:
            outfile.writelines(extracted_data)
        
        print(f"筛选的分数已成功保存到 {output_file} 文件中.")
    
    except Exception as e:
        print(f"出现错误: {e}")

# 使用示例
input_file = 'smutest_netr_fets.log'  # 替换为你的输入文件路径
output_file = 'smunetr_fets.txt'  # 替换为你想保存的输出文件路径

extract_scores(input_file, output_file)

