# Intro
This is repository for Ajou Univ research project: Improving anomaly data classification performance of AnoGAN via improved anomaly score computation using object size.<br/><br/>If you want to see more details, please watch video in Ajou Univ Softcon.

# How to run
anogan.py: this is code for AnoGAN model<br/>
main_re.py: this is for calculating and storing residual score and discrimination score<br/>
test.py: in this file, you can see how anomaly scores are distributed and their performance<br/>
<br/>
So, basically, you can run main_re.py (code edit should be done according to your purpose) and use test.py to see scores distribution and performance.<br/>
<br/>
other files are saved scores and model weights
<br/>
## Caution: 
Some files are not uploaded because of solving some kind of compatibility problem and file size problem.
<br/>I removed all supplementary and unnecessary codes for better readability. However it may (or not) cause malfunction. In that case, please make issue
<br/>Also, some codes are outdated than used for presentation and paper. If you are interested, please contact me personally.
## Reference
AnoGAN model(Code, keras): https://github.com/tkwoo/anogan-keras
