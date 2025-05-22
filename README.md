# LTE-Coverage-Prediction-Using-Machine-Learning
This project develops a machine learning framework to predict LTE (4G) signal quality metrics—RSRP, RSRQ, and SINR—using models trained on suburban and urban environments. It evaluates and compares Random Forest, Gradient Boosting, SVR, and Extra Trees for accurate signal prediction.




Summary of the project: 
----------------------------                              -------------------------------------                           -------------------------------------                        -----------------------------------
Accurately predicting LTE coverage has become more important in recent years due to expansion of mobile data demand. Ensuring stable network performance is essential. Older models, like empirical and deterministic techniques, usually fail, especially in complex environments. They tend to struggle with flexibility and processing demands. This study solves this problem by building a machine learning system designed to predict
radio signal metrics such as RSRP, RSRQ, and SINR, on urban and suburban areas. This project tested different model like Random Forest (RF), Gradient Boosted Trees (GBT), Extremely Randomised Trees (ERTR), and Support Vector Regression (SVR) to compare their performance. Among them, ERTR delivered the best results. Essential predictors included distance(miles), antenna height(meters), and azimuth(degree), identified through feature engineering. The final optimised model was deployed in a user-friendly web app that support real-time batch predictions. Despite its success, it still faces challenges related to generalisation and environmental variability. Future improvements may include data augmentation and transitioning to 5G/6G compatibility. This work offers a practical, scalable alternative to traditional prediction coverage models, shortness the gap between academic theory and deployment ready tools.


# Try the app here 👉 [Open Streamlit App]  (https://ue-lte-coverage-prediction.streamlit.app/)

