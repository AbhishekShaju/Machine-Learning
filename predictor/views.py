from django.shortcuts import render
import pickle
import numpy as np

# Load trained models and encoders
slr_model = pickle.load(open("slr_model.pkl", "rb"))
mlr_model = pickle.load(open("mlr_model.pkl", "rb"))
le_edu, le_job, le_loc = pickle.load(open("encoders.pkl", "rb"))

def slr_prediction(request):
    if request.method == 'POST':
        experience = float(request.POST['experience'])
        salary = round(slr_model.predict(np.array([[experience]]))[0], 2)
        return render(request, 'slr.html', {'salary': salary})
    return render(request, 'slr.html')

def mlr_prediction(request):
    if request.method == 'POST':
        experience = float(request.POST['experience'])
        education = request.POST['education']
        job_role = request.POST['jobrole']
        location = request.POST['location']

        education_encoded = le_edu.transform([education])[0]
        job_role_encoded = le_job.transform([job_role])[0]
        location_encoded = le_loc.transform([location])[0]

        input_data = np.array([[experience, education_encoded, job_role_encoded, location_encoded]])
        salary = round(mlr_model.predict(input_data)[0], 2) 

        return render(request, 'mlr.html', {'salary': salary})
    
    return render(request, 'mlr.html')
