from fly import route, run, request, response, send_file, abort


@route('/')
def hello_world():
  return 'Hello World!'

@route('/hello/:name')
def hello_name(name):
  return 'Hello %s!' % name

@route('/hello', method='POST')
def hello_post():
  name = request.POST['name']
  return 'Hello %s!' % name

@route('/static/:filename#.*#')
def static_file(filename):
  send_file(filename, root='/path/to/static/files/')

if __name__=='__main__':
  run(host='127.0.0.1', port=8000)
