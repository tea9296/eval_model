import akasha
import akasha.eval as eval

################### fact ################################
# ev = eval.Model_Eval(model="openai:gpt-4o",
#                      question_type="fact",
#                      question_style="single_choice")

# ev.auto_create_questionset(
#     "docs/MIC電子商務",
#     100,
# )
#########################################################
#
#
################## summary ##############################
# ev = eval.Model_Eval(model="openai:gpt-4o",
#                      question_type="summary",
#                      question_style="essay")

# ev.auto_create_questionset(
#     "docs/MIC電子商務",
#     100,
# )
#########################################################
#
#
#################### compared ###########################
ev = eval.Model_Eval(model="openai:gpt-4o",
                     question_type="compared",
                     question_style="essay")

ev.auto_create_questionset(
    "docs/MIC電子商務",
    5,
    output_file_path="questionset/MIC電子商務-compared_new5-essay.txt")

#########################################################
#
#
#################### irrelevant ###########################
# ev = eval.Model_Eval(model="openai:gpt-4o",
#                      question_type="irrelevant",
#                      question_style="single_choice")

# ev.auto_create_questionset(
#     "docs/MIC電子商務",
#     100,
#     output_file_path="questionset/MIC電子商務-irrelevant-single_choice.txt")

#########################################################
