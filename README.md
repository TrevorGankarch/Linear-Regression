# Linear-Regression
This is an application of Machine Learning Evaluation of medical data on heart health by an age group sourced from Kaggle. The Linear Regression model is used on the data presented to ensure the analysis is accurate.  
Cleaned the data upon processing it some of the column fields had NaN values in them as 
well as converting some of the strings to float in the heart_disease.csv file. LabelEncoder ()  to convert values in string format to floating point type. NaN stands for Not a Number. Another method is to drop the rows or specific rows contianing NaN blank spaces using dropna() method. 

Also contained in the Logistic Regression algorithm is the hyperparameter tuning method to fine tune Logistic Regression algorithm. 
1. Regularization Strenght represented by C: This paramter is for the controls between the balance between model and complexity and fit. Larger the C value results in a stronger regularization effect leading to small coefficients and simpler model. 
2. Penalty : This technique is used to prevent overfitting by adding a penalty term to the loss function that is proportional to the sum of the squares of the model's coefficients. Basically the amount of shrinkage , where data values are shrunk towards the central point , like the mean. 
3. Solver : It is a linear classification that supports logistic regression and linear support vector machines. The solver uses a Coordinate Descent algorithm that solves optimization problems by successively performaing approximate minimization along coordinate directions or coordinate hyperplanes. It is an algorithm that is used to update the paramters (wieghts and biases) of the model. 
4. Max Iteration: This specifies the default maximum number of iterations. The solver iterates until converges,

So the Logistic Algorithm is a supervised machine learning algorithm used for classification tasks where the goal is to predict the probability that an instance belongs to a given class or not. LR in short is a statiscal algorithm which analyze that relationship between two data factors. There a 3 types of LR , namely bionomial , Multinomial and Ordinal. 

Also when choosing your X independent values and y - dependent value make sure the y value is the one that depends on the outcomes of the analysis on the X independent values of the features in the column in the csv file. 

Logistic Regression is used in real world applications such as predicting an incoming email if it is a spam or not a spam. In health care logistic regression can be used to predict if a tumor is likely to be benign or malignant. In the financial industry it can be used to predict if a transactoin is fradulent ot not. ID Credit scoring in financial companies have predictive models to be easily interpretable. Famous hotel booking app Booking.com has a lot of machine learning methods literally on its website. Gaming with speed is one of the advantages of logistic regression , and it is extremely useful in the gaming industry. 

In Papua New Guinea a research team had used it its research on severe anemia available on Research gate here is the link https://search.app/XHE4yC2eV8mgK3Un6

// As I develop and expand on Logistic Regression in Real world application I will update the code feel free to drop me an email on torretofast6@gmail.com on further work or colloboration or if you want to utilize the algorithm in your line of work or business. 
