#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../build/src')
import libamunmt as nmt

from bottle import request, Bottle, abort

app = Bottle()


@app.route('/translate')
def handle_websocket():
        wsock = request.environ.get('wsgi.websocket')
        if not wsock:
            abort(400, 'Expected WebSocket request.')

        while True:
            try:
                message = wsock.receive()
                if message is not None:
                    trans = nmt.translate(message.split('\n'))
                    wsock.send('\n'.join(trans))
            except WebSocketError:
                break


def parse_args():
    """ parse command arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", dest="config")
    parser.add_argument('-p', dest="port", default=8080, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    nmt.init("-c {}".format(args.config))

    from gevent.pywsgi import WSGIServer
    from geventwebsocket import WebSocketError
    from geventwebsocket.handler import WebSocketHandler
    server = WSGIServer(("0.0.0.0", args.port), app,
                        handler_class=WebSocketHandler)
    server.serve_forever()
