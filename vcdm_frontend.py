import uuid
import pandas as pd
from flask_setup import app
from flask import send_file
from engine.utility import *
from vcdm_backend import vcdm_calc
from gevent.pywsgi import WSGIServer
from flask import flash, request, redirect, render_template, session


IP = '127.0.0.1'
PORT = 5000


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':

        if request.form.getlist('check'):

            # Agreed terms
            session['agree'] = True

            # Delete temporary files before new session starts
            delete_old_files('60')

            # Go the the next page
            return redirect('/input')

        else:

            # Error if extension is not csv.
            flash('You need to agree the terms and conditions.')

            # Repeat the smae page
            return redirect(request.url)


@app.route('/input', methods=['POST', 'GET'])
def upload_file():
    '''First page at the front end. This function supports the fle uploading
    procedure.

    Args:
        None

    Returns:
       Flask redirection, depending on sucessfull uploading. Otherwise error.

    '''

    if request.method == 'GET':

        # Check agreed terms
        if 'agree' in session:
            return render_template('input.html')

        else:
            return redirect('/')

    if request.method == 'POST':

        # Submitted file by the form
        file = request.files['file']

        if file and allowed_file(file.filename):

            # Create a unique filname for avoiding duplicates
            filename = os.path.join(app.config['UPLOAD_FOLDER'],
                                    str(uuid.uuid4()) + '.csv')

            # Upload the file with the new filename
            file.save(filename)

            # Create a few session variables, containing the filename
            session['fname'] = filename

            # Go the the next page
            return redirect('/model')

        else:

            # Error if extension is not csv.
            flash('Please upload a csv file.')

            # Repeat the smae page
            return redirect(request.url)


@app.route('/model', methods=['POST', 'GET'])
def input():
    '''2nd page on the frontend, providing the describing statistics,
    the histogram and the input parameter reader.

    Args:
        None

    Returns:
       Flask redirection, depending on sucessfull uploading. Otherwise error.

    '''

    if request.method == 'GET':

        # Check agreed terms
        if 'agree' in session:

            try:

                # Read the raw data
                df = pd.read_csv(session['fname'])

                # Generate describing statistics
                stats = generate_stats(df)

                # Create the first histogram
                script, div = generate_histogram(df)

                # Generate some additional plots based on column 2 and 4
                script_t, div_t = generate_turbidity_and_flow(df)

                return render_template('model.html', stats=stats,
                                       hist_script=script,
                                       hist_div=div,
                                       turb_script=script_t,
                                       turb_div=div_t)
            except Exception as e:

                # Error message
                err = 'An error occurred while processing your request: '
                flash(err + str(e))

                # Return to the parameter page
                return redirect('/output')

        else:
            return redirect('/')

    if request.method == 'POST':

        # Generate labels for the parameter list
        par_list = ['par' + str(x) for x in range(1, 8)]

        # Convert parameters from string to float if possible
        try:
            session['parameters'] = [float(request.form[x]) for x in par_list]

        # Error if string conversion fails
        except ValueError:

            # Error message to the front end
            flash('Only numbers are allowed as input!')

            # Redirect to the same page if type conversion fails
            return redirect(request.url)

        # Redirect to the next page if parameters are ok
        return redirect('/output')


@app.route('/output', methods=['POST', 'GET'])
def output():
    '''Output page for the fontend, providing the results of the vcdn
    calculations.

    Args:
        None

    Returns:
       Flask redirection, depending on sucessfull uploading. Otherwise error.

    '''

    if request.method == 'GET':

        # Check agreed terms
        if 'agree' in session:

            try:

                # Read the input file
                df = pd.read_csv(session['fname'])

                # Call the model
                re = vcdm_calc(df, session['parameters'])

                # Create the three plots
                o_script, o_div = generate_results(session['fname'], re, df)

                # Generate the results in xls format
                session['xls'] = generate_xls(session['fname'], re, df)

                # Create the first histogram
                script, div = generate_histogram(df)

                return render_template('output.html',
                                       output_script=o_script,
                                       output_div=o_div)

            except KeyboardInterrupt as e:

                # Error message
                err = 'An error occurred while processing your request: '
                flash(err + str(e))

                # Return to the parameter page
                return redirect('/input')

        else:
            return redirect('/')

    if request.method == 'POST':

        if 'download_xlsx' in request.form:

            return send_file(session['xls'], as_attachment=True)


if __name__ == "__main__":

    #app.run(debug=True)
    # Use gevent WSGI server and start the server
    WSGIServer((IP, PORT), app.wsgi_app).serve_forever()
