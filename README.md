# 2026_Machine-Learning_HW1

1. Objective
This assignment focuses on Agentic Software Development. You will act as a manager, orchestrating
AI agents within the Google Antigravity IDE to plan, code, and evaluate a stock price prediction
system. The goal is to move beyond manual coding and master the art of AI orchestration. You are
expected to demonstrate critical thinking by overseeing the logic of the AI agents, ensuring they
follow financial common sense and academic integrity. This exercise will help you understand the
efficiency gains of using agentic workflows while maintaining human oversight to prevent logical
errors like look-ahead bias.
2. Task Requirements
 Environment: Must use Google Antigravity IDE.
 Data Source: Use yfinance to fetch S&P 500 (^GSPC) data.
 Period: start='2021-01-01', end='2025-12-31'.
 Algorithms: Compare the following two models: XGBoost、Random Forest
 Evaluation Metric: Mean Squared Error (MSE).
 Data Splitting (Mandatory):
To ensure a fair comparison, you must split the data chronologically with no random shuffling.
Training Set: 2021-01-01 to 2024-12-31.
Testing Set: 2025-01-01 to 2025-12-31.
