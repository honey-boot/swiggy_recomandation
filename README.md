🍽️ Swiggy Restaurant Recommendation System
This project builds a restaurant recommendation system using machine learning techniques like clustering and encoding. It helps users find similar restaurants based on their preferences such as location, cuisine, cost, and rating.

📌 Project Objective
To recommend top restaurants to users based on their inputs like:

City

Cuisine

Rating

Cost

We use clustering to group similar restaurants and recommend options from the same group (cluster).

🧰 Technologies & Libraries Used
Python

Pandas, NumPy

Scikit-learn (LabelEncoder, OneHotEncoder, StandardScaler, KMeans, ColumnTransformer)

Pickle (for saving models)

Streamlit (for frontend, if integrated)

CSV files for data storage

🧪 Workflow Overview
✅ 1. Data Cleaning
Dropped duplicates and null values.

Removed unwanted columns (link, address, menu, lic_no).

Cleaned the cost, rating, and rating_count columns for numeric use.

Split city into city and main_city.

✅ 2. Encoding
Label Encoded restaurant names (name) for tracking.

One-Hot Encoded categorical columns: city, main_city, cuisine.

✅ 3. Feature Scaling
Scaled features using StandardScaler to normalize cost, rating, etc.

✅ 4. Clustering
Trained a KMeans model to group restaurants into 10 clusters based on similarity.

Saved clusters to make real-time recommendations.

✅ 5. Recommendation Logic
Takes user input (e.g., city, cuisine, rating, cost)

Encodes and scales it the same way as the training data.

Predicts the user's cluster.

Recommends top restaurants from that cluster based on rating and popularity.

📈 Future Improvements
Add a Streamlit UI for interactive user input

Use collaborative filtering or deep learning

Include menu or dish-level recommendations

Integrate with real Swiggy APIs (if available)

