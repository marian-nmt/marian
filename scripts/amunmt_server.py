#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from flask import Flask
from flask_sockets import Sockets
import libamunmt as nmt

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
            translation = nmt.translate(inList)
            ws.send(translation)

if __name__ == '__main__':
    from gevent import pywsgi
    from geventwebsocket.handler import WebSocketHandler

    server = pywsgi.WSGIServer(('', int(sys.argv[2])), app,
                               handler_class=WebSocketHandler)
    print >> sys.stderr, "Server is running"
    server.serve_forever()
