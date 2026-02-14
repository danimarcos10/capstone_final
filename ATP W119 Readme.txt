PEW RESEARCH CENTER
Wave 119 American Trends Panel 
Dates: Dec. 12, 2022 - Dec. 18, 2022
Mode: Web 
Sample: Full panel
Language: English and Spanish
N=11,004

***************************************************************************************************************************
NOTES

For a small number of respondents with high risk of identification, certain values have been randomly swapped with those of lower risk cases with similar characteristics.

BIASBETR/BIASWORS/BIASSAME/AIWRKH5Y/AIWRKH5N
The W119 dataset contains the following coded open-end responses to BIASBETR, BIASWORS, BIASSAME, AIWRKH5Y, and AIWRKH5N. Up to three mentions were coded for each open end.
BIASBETR_OE1_W119
BIASBETR_OE2_W119
BIASBETR_OE3_W119
BIASWORS_OE1_W119
BIASWORS_OE2_W119
BIASWORS_OE3_W119
BIASSAME_OE1_W119
BIASSAME_OE2_W119
BIASSAME_OE3_W119
AIWRKH5Y_OE1_W119
AIWRKH5Y_OE2_W119
AIWRKH5Y_OE3_W119
AIWRKH5N_OE1_W119
AIWRKH5N_OE2_W119
AIWRKH5N_OE3_W119

AIKNOW1-AIKNOW7
The W119 dataset also includes the following created variables which indicate which knowledge questions from AIKNOW1 through AIKNOW7 were answered correctly, and the total answered correctly. Syntax can be found in the syntax section below. 
AIKNOW1_CORRECT_W119
AIKNOW2_CORRECT_W119
AIKNOW3_CORRECT_W119
AIKNOW5_CORRECT_W119
AIKNOW6_CORRECT_W119
AIKNOW7_CORRECT_W119
AIKNOW_INDEX_W119

INDUSTRYCOMBO
The W119 dataset contains a created variable, INDUSTRYCOMBO. Due to small sample size for analysis, some categories in INDUSTRY were further collapsed into larger net categories as follows: The arts, entertainment and recreation industry was combined with hospitality or service; and the agriculture, forestry, fishing and hunting industry was added to manufacturing, mining, or construction.


***************************************************************************************************************************
WEIGHTS 


WEIGHT_W119 is the weight for the sample. Data for all Pew Research Center reports are analyzed using this weight.


***************************************************************************************************************************
Releases from this survey:

FEBRUARY 15, 2023 "Public Awareness of Artificial Intelligence in Everyday Activities"
https://www.pewresearch.org/science/2023/02/15/public-awareness-of-artificial-intelligence-in-everyday-activities/

FEBRUARY 22, 2023 "How Americans view emerging uses of artificial intelligence, including programs to generate text or art"
https://www.pewresearch.org/short-reads/2023/02/22/how-americans-view-emerging-uses-of-artificial-intelligence-including-programs-to-generate-text-or-art/

FEBRUARY 22, 2023 "60% of Americans Would Be Uncomfortable With Provider Relying on AI in Their Own Health Care"
https://www.pewresearch.org/science/2023/02/22/60-of-americans-would-be-uncomfortable-with-provider-relying-on-ai-in-their-own-health-care/

APRIL 20, 2023 "AI in Hiring and Evaluating Workers: What Americans Think"
https://www.pewresearch.org/internet/2023/04/20/ai-in-hiring-and-evaluating-workers-what-americans-think/

APRIL 20, 2023 "Most Americans say racial bias is a problem in the workplace. Can AI help?"
https://www.pewresearch.org/short-reads/2023/04/20/most-americans-say-racial-bias-is-a-problem-in-the-workplace-can-ai-help/

MAY 17, 2023 "How U.S. adults on Twitter use the site in the Elon Musk era"
https://www.pewresearch.org/short-reads/2023/05/17/how-us-adults-on-twitter-use-the-site-in-the-elon-musk-era/

July 26, 2023 "Which U.S. Workers Are More Exposed to AI on Their Jobs?"
https://www.pewresearch.org/social-trends/2023/07/26/which-u-s-workers-are-more-exposed-to-ai-on-their-jobs/

NOVEMBER 21, 2023 "What the data says about Americans’ views of artificial intelligence"
https://www.pewresearch.org/short-reads/2023/11/21/what-the-data-says-about-americans-views-of-artificial-intelligence/


***************************************************************************************************************************
SYNTAX

**Syntax to create AIKNOW1_CORRECT through AIKNOW7_CORRECT and AIKNOW_INDEX_W119.
AIKNOW_INDEX_W119

**AIKNOW1 – customer service

recode AIKNOW1_W119 (1=1) (else=0) into AIKNOW1_CORRECT_W119.
variable labels AIKNOW1_CORRECT_W119 'Answered AIKNOW1 correctly'.
value labels AIKNOW1_CORRECT_W119
0 'no'
1 'yes'.
formats AIKNOW1_CORRECT_W119 (f1.0).
execute.

**AIKNOW2 – playing music

recode AIKNOW2_W119 (2=1) (else=0) into AIKNOW2_CORRECT_W119.
variable labels AIKNOW2_CORRECT_W119 'Answered AIKNOW2 correctly'.
value labels AIKNOW2_CORRECT_W119
0 'no'
1 'yes'.
formats AIKNOW2_CORRECT_W119 (f1.0).
execute.

**AIKNOW3 - email

recode AIKNOW3_W119 (3=1) (else=0) into AIKNOW3_CORRECT_W119.
variable labels AIKNOW3_CORRECT_W119 'Answered AIKNOW3 correctly'.
value labels AIKNOW3_CORRECT_W119
0 'no'
1 'yes'.
formats AIKNOW3_CORRECT_W119 (f1.0).
execute.

**AIKNOW5 – health products

recode AIKNOW5_W119 (1=1) (else=0) into AIKNOW5_CORRECT_W119.
variable labels AIKNOW5_CORRECT_W119 'Answered AIKNOW5 correctly'.
value labels AIKNOW5_CORRECT_W119
0 'no'
1 'yes'.
formats AIKNOW5_CORRECT_W119 (f1.0).
execute.

**KNOW6 – online shopping

recode AIKNOW6_W119 (3=1) (else=0) into AIKNOW6_CORRECT_W119.
variable labels AIKNOW6_CORRECT_W119 'Answered AIKNOW6 correctly'.
value labels AIKNOW6_CORRECT_W119
0 'no'
1 'yes'.
formats AIKNOW6_CORRECT_W119 (f1.0).
execute.

**KNOW7 – home devices

recode AIKNOW7_W119 (2=1) (else=0) into AIKNOW7_CORRECT_W119.
variable labels AIKNOW7_CORRECT_W119 'Answered AIKNOW7 correctly'.
value labels AIKNOW7_CORRECT_W119
0 'no'
1 'yes'.
formats AIKNOW7_CORRECT_W119 (f1.0).
execute.

**create index

compute AIKNOW_INDEX_W119 = AIKNOW1_CORRECT_W119 + AIKNOW2_CORRECT_W119 + AIKNOW3_CORRECT_W119 + AIKNOW5_CORRECT_W119 + AIKNOW6_CORRECT_W119 + AIKNOW7_CORRECT_W119.
variable labels AIKNOW_INDEX_W119 'Total number of correct AI knowledge answers 0 to 6'.
formats AIKNOW_INDEX_W119 (f1.0).
execute.




