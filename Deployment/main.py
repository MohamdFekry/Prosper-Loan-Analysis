from flask import Flask,render_template,request
import numpy as np
import pickle
modelEMI = pickle.load(open('pipeEMI.pkl','rb'))
modelELA = pickle.load(open('pipeELA.pkl','rb'))
modelPROI = pickle.load(open('pipePROI.pkl','rb'))
model2 = pickle.load(open('pipenew2.pkl','rb'))

app = Flask(__name__,template_folder='templates')

@app.route('/')
def index():
    return render_template('new.html')

@app.route('/predict',methods=['POST'])
def predict_variables():
    ProsperScore = float(request.form.get('ProsperScore'))
    Term = int(request.form.get('Term'))
    BorrowerRate = float(request.form.get('BorrowerRate'))
    ProsperRating = request.form.get('ProsperRating')
    LoanType = int(request.form.get('LoanType'))
    IsBorrowerHomeowner = int(request.form.get('IsBorrowerHomeowner'))
    CurrentlyInGroup = int(request.form.get('CurrentlyInGroup'))
    OpenRevolvingAccounts = int(request.form.get('OpenRevolvingAccounts'))
    OpenRevolvingMonthlyPayment = float(request.form.get('OpenRevolvingMonthlyPayment'))
    IncomeRange = request.form.get('IncomeRange')
    IncomeVerifiable = int(request.form.get('IncomeVerifiable'))
    StatedMonthlyIncome = float(request.form.get('StatedMonthlyIncome'))
    LoanOriginalAmount = int(request.form.get('LoanOriginalAmount'))
    MonthlyLoanPayment = float(request.form.get('MonthlyLoanPayment'))
    LP_CustomerPayments = float(request.form.get('LP_CustomerPayments'))
    LP_CustomerPrincipalPayments = float(request.form.get('LP_CustomerPrincipalPayments'))
    LP_InterestandFees = float(request.form.get('LP_InterestandFees'))
    LP_ServiceFees = float(request.form.get('LP_ServiceFees'))
    LP_CollectionFees = float(request.form.get('LP_CollectionFees'))
    LP_GrossPrincipalLoss = float(request.form.get('LP_GrossPrincipalLoss'))
    LP_NetPrincipalLoss = float(request.form.get('LP_NetPrincipalLoss'))
    Investors = int(request.form.get('Investors'))
    BorrowerAPR = float(request.form.get('BorrowerAPR'))
    BorrowerState = request.form.get('BorrowerState')
    Occupation = request.form.get('Occupation')
    EmploymentStatus = request.form.get('EmploymentStatus')
    CreditScoreRangeLower = float(request.form.get('CreditScoreRangeLower'))
    CreditScoreRangeUpper = float(request.form.get('CreditScoreRangeUpper'))
    TotalCreditLinespast7years = float(request.form.get('TotalCreditLinespast7years'))
    TotalInquiries = float(request.form.get('TotalInquiries'))
    CurrentDelinquencies = float(request.form.get('CurrentDelinquencies'))
    DelinquenciesLast7Years = float(request.form.get('DelinquenciesLast7Years'))
    PublicRecordsLast10Years = float(request.form.get('PublicRecordsLast10Years'))
    EmploymentStatusDuration = float(request.form.get('EmploymentStatusDuration'))
    CurrentCreditLines = float(request.form.get('CurrentCreditLines'))
    OpenCreditLines = float(request.form.get('OpenCreditLines'))
    AmountDelinquent = float(request.form.get('AmountDelinquent'))
    RevolvingCreditBalance = float(request.form.get('RevolvingCreditBalance'))
    BankcardUtilization = float(request.form.get('BankcardUtilization'))
    AvailableBankcardCredit = float(request.form.get('AvailableBankcardCredit'))
    TradesNeverDelinquent = float(request.form.get('TradesNeverDelinquent'))
    DebtToIncomeRatio = float(request.form.get('DebtToIncomeRatio'))
    CreditGrade = request.form.get('CreditGrade')

    # jio=np.array([ProsperScore,Term,BorrowerRate,ProsperRating,LoanType,IsBorrowerHomeowner,CurrentlyInGroup,OpenRevolvingAccounts,OpenRevolvingMonthlyPayment,IncomeRange,IncomeVerifiable,StatedMonthlyIncome,LoanOriginalAmount,MonthlyLoanPayment,LP_CustomerPayments,LP_CustomerPrincipalPayments,LP_InterestandFees,LP_ServiceFees,LP_CollectionFees,LP_GrossPrincipalLoss,LP_NetPrincipalLoss,Investors,BorrowerAPR,BorrowerState,Occupation,EmploymentStatus,CreditScoreRangeLower,CreditScoreRangeUpper,TotalCreditLinespast7years,TotalInquiries,CurrentDelinquencies,DelinquenciesLast7Years,PublicRecordsLast10Years,EmploymentStatusDuration,CurrentCreditLines,OpenCreditLines,AmountDelinquent,RevolvingCreditBalance,BankcardUtilization,AvailableBankcardCredit,TradesNeverDelinquent,DebtToIncomeRatio,CreditGrade])
    # kopa ="#"
    # for i in jio:
    #     kopa+='#'
    #     kopa+=i
    # return kopa
    EMI = modelEMI.predict(np.array([ProsperScore,Term,BorrowerRate,ProsperRating,LoanType,IsBorrowerHomeowner,CurrentlyInGroup,OpenRevolvingAccounts,OpenRevolvingMonthlyPayment,IncomeRange,IncomeVerifiable,StatedMonthlyIncome,LoanOriginalAmount,MonthlyLoanPayment,LP_CustomerPayments,LP_CustomerPrincipalPayments,LP_InterestandFees,LP_ServiceFees,LP_CollectionFees,LP_GrossPrincipalLoss,LP_NetPrincipalLoss,Investors,BorrowerAPR,BorrowerState,Occupation,EmploymentStatus,CreditScoreRangeLower,CreditScoreRangeUpper,TotalCreditLinespast7years,TotalInquiries,CurrentDelinquencies,DelinquenciesLast7Years,PublicRecordsLast10Years,EmploymentStatusDuration,CurrentCreditLines,OpenCreditLines,AmountDelinquent,RevolvingCreditBalance,BankcardUtilization,AvailableBankcardCredit,TradesNeverDelinquent,DebtToIncomeRatio,CreditGrade],dtype=object).reshape(1,43))
    # EMI = modelEMI.predict([[5.950066585742402, 36, 0.158, 'Missing', 0, 1, 1, 1, 24.0,
    #     '$25,000-49,999', 1, 3083.333333, 9425, 330.43, 11396.14, 9425.0,
    #     1971.14, -133.18, 0.0, 0.0, 0.0, 258, 0.16516, 'CO', 'Other',
    #     'Self-employed', 640.0, 659.0, 12.0, 3.0, 2.0, 4.0, 0.0, 2.0,
    #     5.0, 4.0, 472.0, 0.0, 0.0, 1500.0, 0.81, 0.17, 'C']])
    
    # ELA = modelELA.predict([[5.950066585742402, 36, 0.158, 'Missing', 0, 1, 1, 1, 24.0,
    #     '$25,000-49,999', 1, 3083.333333, 9425, 330.43, 11396.14, 9425.0,
    #     1971.14, -133.18, 0.0, 0.0, 0.0, 258, 0.16516, 'CO', 'Other',
    #     'Self-employed', 640.0, 659.0, 12.0, 3.0, 2.0, 4.0, 0.0, 2.0,
    #     5.0, 4.0, 472.0, 0.0, 0.0, 1500.0, 0.81, 0.17, 'C']])
    
    # PROI = modelPROI.predict([[5.950066585742402, 36, 0.158, 'Missing', 0, 1, 1, 1, 24.0,
    #     '$25,000-49,999', 1, 3083.333333, 9425, 330.43, 11396.14, 9425.0,
    #     1971.14, -133.18, 0.0, 0.0, 0.0, 258, 0.16516, 'CO', 'Other',
    #     'Self-employed', 640.0, 659.0, 12.0, 3.0, 2.0, 4.0, 0.0, 2.0,
    #     5.0, 4.0, 472.0, 0.0, 0.0, 1500.0, 0.81, 0.17, 'C']])
    ELA = modelELA.predict(np.array([ProsperScore,Term,BorrowerRate,ProsperRating,
        LoanType,IsBorrowerHomeowner,CurrentlyInGroup,
       OpenRevolvingAccounts,OpenRevolvingMonthlyPayment,IncomeRange,
       IncomeVerifiable,StatedMonthlyIncome,LoanOriginalAmount,
       MonthlyLoanPayment,LP_CustomerPayments,
       LP_CustomerPrincipalPayments,LP_InterestandFees,LP_ServiceFees,
       LP_CollectionFees,LP_GrossPrincipalLoss,LP_NetPrincipalLoss,
       Investors,BorrowerAPR,BorrowerState,Occupation,
        EmploymentStatus,CreditScoreRangeLower,CreditScoreRangeUpper,
       TotalCreditLinespast7years,TotalInquiries,CurrentDelinquencies,
       DelinquenciesLast7Years,PublicRecordsLast10Years,
       EmploymentStatusDuration,CurrentCreditLines,OpenCreditLines,
       AmountDelinquent,RevolvingCreditBalance,BankcardUtilization,
        AvailableBankcardCredit,TradesNeverDelinquent,
       DebtToIncomeRatio,CreditGrade]).reshape(1,43))
    # result2 = model2.predict([[5.950066585742402, 36, 0.158, 'Missing', 0, 1, 1, 1, 24.0,
    #     '$25,000-49,999', 1, 3083.333333, 9425, 330.43, 11396.14, 9425.0,
    #     1971.14, -133.18, 0.0, 0.0, 0.0, 258, 0.16516, 'CO', 'Other',
    #     'Self-employed', 640.0, 659.0, 12.0, 3.0, 2.0, 4.0, 0.0, 2.0,
    #     5.0, 4.0, 472.0, 0.0, 0.0, 1500.0, 0.81, 0.17, 'C']])
    
    PROI = modelPROI.predict(np.array([ProsperScore,Term,BorrowerRate,ProsperRating,
        LoanType,IsBorrowerHomeowner,CurrentlyInGroup,
       OpenRevolvingAccounts,OpenRevolvingMonthlyPayment,IncomeRange,
       IncomeVerifiable,StatedMonthlyIncome,LoanOriginalAmount,
       MonthlyLoanPayment,LP_CustomerPayments,
       LP_CustomerPrincipalPayments,LP_InterestandFees,LP_ServiceFees,
       LP_CollectionFees,LP_GrossPrincipalLoss,LP_NetPrincipalLoss,
       Investors,BorrowerAPR,BorrowerState,Occupation,
        EmploymentStatus,CreditScoreRangeLower,CreditScoreRangeUpper,
       TotalCreditLinespast7years,TotalInquiries,CurrentDelinquencies,
       DelinquenciesLast7Years,PublicRecordsLast10Years,
       EmploymentStatusDuration,CurrentCreditLines,OpenCreditLines,
       AmountDelinquent,RevolvingCreditBalance,BankcardUtilization,
        AvailableBankcardCredit,TradesNeverDelinquent,
       DebtToIncomeRatio,CreditGrade]).reshape(1,43))
    
    result2 = model2.predict(np.array([ProsperScore,Term,BorrowerRate,ProsperRating,
        LoanType,IsBorrowerHomeowner,CurrentlyInGroup,
       OpenRevolvingAccounts,OpenRevolvingMonthlyPayment,IncomeRange,
       IncomeVerifiable,StatedMonthlyIncome,LoanOriginalAmount,
       MonthlyLoanPayment,LP_CustomerPayments,
       LP_CustomerPrincipalPayments,LP_InterestandFees,LP_ServiceFees,
       LP_CollectionFees,LP_GrossPrincipalLoss,LP_NetPrincipalLoss,
       Investors,BorrowerAPR,BorrowerState,Occupation,
        EmploymentStatus,CreditScoreRangeLower,CreditScoreRangeUpper,
       TotalCreditLinespast7years,TotalInquiries,CurrentDelinquencies,
       DelinquenciesLast7Years,PublicRecordsLast10Years,
       EmploymentStatusDuration,CurrentCreditLines,OpenCreditLines,
       AmountDelinquent,RevolvingCreditBalance,BankcardUtilization,
        AvailableBankcardCredit,TradesNeverDelinquent,
       DebtToIncomeRatio,CreditGrade]).reshape(1,43))
    
    lis = []
    lis.append(EMI[0])
    lis.append(ELA[0])
    lis.append(PROI[0])
    ris = int(result2[0])
    if(ris==1):
        lis.append('Low Risk')
    elif(ris==2):
        lis.append('Moderate Risk')
    elif(ris==3):
        lis.append('High Risk')
    else:
        lis.append('Very High Risk')

    post = {
        'emi':str(lis[0]),
        'ela':str(lis[1]),
        'Proi':str(lis[2]),
        'rik':lis[3]


    }

    # result=str(EMI[0])+" "+str(ELA[0])+" "+str(PROI[0])+" "+lis[3]
    # res = "kkk"
    return render_template('output.html', result=post)

    

if __name__=='__main__':
    app.run(debug = True)
