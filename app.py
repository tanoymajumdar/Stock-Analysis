from flask import Flask, request, render_template, Response
import arrano
import logging
import pandas as pd
import sys
import json
import io
#import StringIO
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


app = Flask(__name__, template_folder='templates')

@app.route('/raw', methods=['GET', 'POST'])
def rawData():
    commodity = request.form['comname'] 
    crypto = request.form['cryname'] 
    start = request.form['startday']
    end = request.form['endday']
    interval = request.form['interval']
    cor_type = request.form['cor_type']
    arrano.correlations(commodity, crypto, start, end, interval, cor_type).savefig('static/price.png')
    arrano.returns_graph(commodity, crypto, start, end, interval, cor_type).savefig('static/returns.png')
    return render_template("kviewData.html", column_names=df_commodity.columns.values, column_names_1=df_crypto.columns.values, row_data=list(df_commodity.values.tolist()), row_data_1=list(df_crypto.values.tolist()), zip = zip)
    #return "Hello"

@app.route('/graphs.png', methods=['GET', 'POST'])
def graphPlot():
    commodity = request.form['comname'] 
    crypto = request.form['cryname'] 
    start = request.form['startday']
    end = request.form['endday']
    interval = request.form['interval']
    plt_commodity = arrano.plotGraphs_commodity(commodity, start, end, interval)
    plt_commodity.savefig('/static/new_plot.png')
    return render_template('plotting.html', name = 'new_plot', url ='/static/new_plot.png')
    """
    canvas = FigureCanvas(plt_commodity)
    output = StringIO.StringIO()
    canvas.print_png(output)
    response = make_response(output.getvalue())
    response.mimetype = 'image/png'
    return response
"""
