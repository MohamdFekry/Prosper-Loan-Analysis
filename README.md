#Loan Prediction Flask App
This is a Flask web application that allows users to predict loan-related variables based on their inputs using machine learning models. Specifically, the app provides predictions for three different models: one for the estimated monthly installment (EMI), another for the estimated loan amount (ELA), and a third for the estimated probability of default (PROI).

##Table of Contents
Installation
Usage
Models
Data
Endpoints

##Installation
To run the app, you will need to have Python 3.7 or higher installed on your system. You can download Python from the official website.

Once you have Python installed, you can clone this repository to your local machine by running the following command in your terminal:

bash
Copy code
	git clone https://github.com/yourusername/loan-prediction-flask-app.git
Then, navigate to the root directory of the project and install the required packages by running:

bash
Copy code
	pip install -r requirements.txt

##Usage
To start the app, simply run the following command in your terminal from the root directory of the project:

bash
Copy code
	python app.py
This will start the Flask server on your local machine. You can then access the app by opening a web browser and navigating to http://localhost:5000/.

##Models
The app uses three different machine learning models to predict loan-related variables. The models were trained on a dataset of historical loan data using the following techniques:

	EMI, ELA, PROI: xgboostregressor
	Risk status: xgboostclassifier
The models were saved as Python pickle files (*.pkl) and are loaded into memory when the app starts.

##Data
The dataset used to train the machine learning models is not included in this repository due to its large size. However, a sample of the data can be found in the data directory. The dataset contains the following columns:

	ProsperScore
	Term
	BorrowerRate
	ProsperRating
	LoanType
	IsBorrowerHomeowner
	CurrentlyInGroup
	OpenRevolvingAccounts
	OpenRevolvingMonthlyPayment
	IncomeRange
	IncomeVerifiable
	StatedMonthlyIncome
	LoanOriginalAmount
	MonthlyLoanPayment
	LP_CustomerPayments
	LP_CustomerPrincipalPayments
	LP_InterestandFees
	LP_ServiceFees
	LP_CollectionFees
	LP_GrossPrincipalLoss
	LP_NetPrincipalLoss
	Investors
	BorrowerAPR
	BorrowerState
	Occupation
	EmploymentStatus
	CreditScoreRangeLower
	CreditScoreRangeUpper
	TotalCreditLinespast7years
	TotalInquiries
	CurrentDelinquencies
	DelinquenciesLast7Years
	PublicRecordsLast10Years
	EmploymentStatusDuration
	CurrentCreditLines
	OpenCreditLines
	AmountDelinquent
	RevolvingCreditBalance
	BankcardUtilization
	AvailableBankcardCredit
	TradesNeverDelinquent
	DebtToIncomeRatio
	CreditGrade

##Endpoints
The app provides the following endpoints:
	/: the main page of the app, where users can enter their inputs and get predictions.
