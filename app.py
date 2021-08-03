from flask import Flask, render_template,url_for,request
import numpy as np
from pandas import DataFrame
import pickle




app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('index.html')



@app.route('/resultat', methods=['POST', 'GET'])
def predict():
    mt_tran = float(request.form['montant_transaction'])
    nbr_tran = float(request.form['nbre_transaction'])
    chgt_tran = float(request.form['chgt_transaction'])
    chgt_mt = float(request.form['chgt_montant'])
    pd_mens = float(request.form['produit_mensuel'])
    mois_inac = float(request.form['mois_inactivite'])
    nbre_cont = float(request.form['nbre_contact'])
    solde = float(request.form['solde_renouv'])
    taux_util = float(request.form['taux_utilisation'])

    liste = [mt_tran, nbr_tran, chgt_tran, chgt_mt, pd_mens, mois_inac, nbre_cont, solde, taux_util]
    # model = pickle.load(open('model.pkl', 'rb'))
    f = open('vrai_model.sav', "rb")

    mdl = pickle.load(f)

    f.close()

    def prediction(X):
        y_pred = mdl.predict(X)
        return (y_pred)

    liste = np.transpose(DataFrame(liste))
    def resultat_prediction():
        a = prediction(liste) #passer ma liste a la fonction de prédiction
        a = str(a).strip('[]') #convertion du resultat en sting et retrais des cotes []
        a = int(a)            #convertion sdu resultat en int pour pouvoir faire la comparaison
        return a
    pred = resultat_prediction()

    def statement():
        if pred == 0:
            return 'Résultat :  Félicitation ! Ce client est succeptible de rester.'
        elif pred == 1:
            return 'Résultat : Désolé ! Ce client est succeptible de partir.'

    return render_template('index.html', statement=statement())






    '''def resultat(X):
        d = np.transpose(DataFrame(X))
        y_pred = mdl.predict(X)
        a = y_pred(d)
        print(str(a).strip('[]'))'''
    
    '''resultat(liste)'''
    # def statement():
    #      if pred == 0:
    #          return 'Result:- The model has predicted that you will not suffer from any cardic arresst but you should take care of your self.'
    #      elif pred == 1:
    #          return 'Result:- You should consult with doctor, The model'
   
    # return render_template('resultat.html', statement=statement())

if __name__ == '__main__':
    app.run(debug=True) 