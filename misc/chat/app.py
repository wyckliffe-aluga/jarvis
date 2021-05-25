from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from gevent.pywsgi import WSGIServer

app = Flask(__name__) 

@app.route('/bot', methods=['POST'])
def bot():
    incoming_msg = request.values['Body']

    # use the incoming message to generate the response here

    r = MessagingResponse()
    r.message('this is the response')
    return str(r)

http_server = WSGIServer(('', 5000), app)
http_server.serve_forever()