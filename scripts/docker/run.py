import sys
import libamunmt as nmt
from flask import Flask
from flask_sockets import Sockets

reload(sys)
sys.setdefaultencoding('utf8')

app = Flask(__name__)
sockets = Sockets(app)

print >> sys.stderr, sys.argv

nmt.init('-c {}'.format(sys.argv[1]))

@sockets.route('/translate')
def translate(ws):
    while not ws.closed:
        message = ws.receive()

        if message:
            inList = message.split("\n")
            print ("Log input to MT: *" + message + "*")
            translation = nmt.translate(inList)
            print ("Log output from MT: *" + "".join(translation) + "*")
            ws.send(format_response(translation))

def format_request(input):
    sentences = []
    for line in input:
        sentences.append(line)
    return sentences

def format_response(output):
    return {
        'result': {
            'hypotheses': [
                {'translation': translation} for translation in output
            ],
        },
    }


if __name__ == '__main__':
    from gevent import pywsgi
    from geventwebsocket.handler import WebSocketHandler

    print >> sys.stderr, "Server is running"
    server = pywsgi.WSGIServer(('', 8080), app, handler_class=WebSocketHandler)
    server.serve_forever()
