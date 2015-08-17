#!/usr/local/bin/python3

"""
-------

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

    run(host='localhost', port=8000)

"""

__author__ = 'Taku Kawasaki'
__version__ = ('0', '1')
__license__ = 'MIT'


import cgi
import mimetypes
import os
import os.path
import sys
import traceback
import re
import random
import threading
import time

try:
  import http.cookies
except ImportError:
  import Cookie

  
try:
  from urllib.parse import parse_qs
except ImportError:
  from cgi import parse_qs
    

try:
  import pickle as pickle
except ImportError:
  import pickle

try:
  import dbm as dbm
except ImportError:
  import dbm.ndbm


class FlyException(Exception):
  """A base class for exception used by fly"""
  pass

class HTTPError(FlyException):
  """A way to break the execution and instantly jump to ab error handler"""

  def __init__(self,status,text):
    self.output = text
    self.http_status = int(status)

  def __str__(self):
    return self.output


  
class BreakTheFly(FlyException):
  """not an exception ,jump out of controll code.
  WSGI Handler to instantly call start_response()"""

  def __init__(self,output):
    self.output = output



    
class TemplateError(FlyException):
  """error exceptin in template engine"""
  pass




def WSGIHandler(environ,start_response):
  """WSGI Handler"""
  global request
  global response
  request.bind(environ)
  response.bind()

  try:
    handler,args = match_url(request.path,request.method)
    if not handler:
      raise HTTPError(404,"Not found")

    output = handler(**args)
  except BreakTheFly as shard:
    output = shard.output

  except Exception as exception:
    response.status = getattr(exception,'http_status',500)
    errorhandler = ERROR_HANDLER.get(response.status,None)
    if errorhandler:
      try:
        output = errorhandler(exception)
      except:
        output = "Exception within error handler! Application stopped"
    else:
      if DEBUG:
        output = "Exception {!s}: {!s}".format(exception.__class__.__name__,
                                               str(exception))

      else:
        output = "Unhandled exception: Application stopped."

    if response.status == 500:
      request._environ['wsgi.errors'].write("Error (500) on '{!s}': {!s}\n".format(request.path,exception))

  db.close()

  if hasattr(output,'read'):
    fileoutput = output
    if 'wsgi.file_wrapper' in environ:
      output = environ['wsgi.file_wrapper'](fileoutput)
    else:
      output = iter(lambda: fileoutput.read(8192),'')

  elif isinstance(output,str):
    output = [output]

  for c in list(response.COOKIES.values()):
    response.header.add('Set-Cookie', c.OutputString())

  status = '{:d} {!s}'.format(response.status,HTTP_CODES[response.status])
  start_response(status,list(response.header.items()))
  return output

class Request(threading.local):
  """Represents a single request using thread-local namespace"""

  def bind(self ,environ):
    self._environ =  environ
    self._GET = None
    self._POST = None
    self._GETPOST = None
    self._COOKIES = None

    self.path = self._environ.get('PATH_INFO', '/').strip()
    if not self.path.startswith('/'):
      self.path = '/' + self.path

  @property
  def method(self):
    """Returns the request method (GET ,POST,PUT,DELETE,...)"""
    return self._environ.get('REQUEST_METHOD','GET').upper()

  @property
  def query_string(self):
    """content of query_string"""
    return self._environ.get('QUERY_STRING','')

  @property
  def input_length(self):
    """Content of CONTENT_LENGTH"""
    try:
      return int(self._environ.get('CONTENT_LENGTH','0'))
    except ValueError:
      return 0


  @property
  def GET(self):
    """Returns a dict with GET parameters"""
    if self._GET is None:
      raw_dict = parse_qs(self.query_string,keep_blank_values=1)
      self._GET = {}
      for key,value in list(raw_dict.items()):
        if len(value) == 1:
          self._GET[key] = value[0]

        else:
          self_GET[key] = value
    return self._GET

  @property
  def POST(self):
    """Returns a dict with parsed POST data"""
    if self._POST is None:
      raw_data = cgi.FieldStorage(fp=self._environ['wsgi.input'],environ=self._environ)
      self._POST = {}
      for key in raw_data:
        if raw_data[key].filename:
          self._POST[key] = raw_data[key]

        elif isinstance(raw_data[key],list):
          self._POST[key] = [v.value for v in raw_data[key]]

        else:
          self._POST[key] = raw_data[key].value
    return self._POST

  @property
  def params(self):
    """Returns a mix of GET and POST data. POST overwrites GET"""
    if self._GETPOST is None:
      self._GETPOST = dict(self.GET)
      self._GETPOST.update(self.POST)

  @property
  def COOKIES(self):
    """Returns a dict with COOKIES."""
    if self._COOKIES is None:
      raw_dict = http.cookies.SimpleCookie(
        self._environ.get('HTTP_COOKIE',''))
      self._COOKIES = {}
      for cookie in list(raw_dict.values()):
        self._COOKIES[cookie.key] = cookie.value
    return self._COOKIES


  
  
class Response(threading.local):
  """Represent a single response using thread local namespace"""

  def bind(self):
    """clears old data and creates a brand new response"""

    self._COOKIES = None
    self.status = 200
    self.header = HeaderDict()
    self.content_type = 'text/html'
    self.error = None

  @property
  def COOKIES(self):
    if not self._COOKIES:
      self._COOKIES = http.cookies.SimpleCookie()
    return self._COOKIES

  def set_cookie(self,key,**kargs):
    """sets a cookie. optional settings : expires ,path,commment domain etc"""
    self.COOKIES[key] = value
    for k in kargs:
      self.COOKIES[key][k] = kargs[k]
  def get_content_type(self):
    '''Gives access to the 'Content-Type' header and defaults to 'text/html'.'''
    return self.header['Content-Type']

  def set_content_type(self,value):
    self.header['Content-Type'] = value

  content_type = property(get_content_type,set_content_type,None,get_content_type.__doc__)


class HeaderDict(dict):
  """A dictionary with case insensitive keys."""
  def __setitem__(self,key,value):
    return dict.__setitem__(self,key.title(),value)

  def __getitem__(self,key):
    return dict.__getitem__(self,key.title())

  def __delitem__(self,key):
    return dict.__delitem__(self,key.title())

  def __contains__(self,key):
    return dict.__contains__(self,key.title())


  def items(self):
    for key ,values in dict.items(self):
      if not isinstance(values, list):
        values = [values]
      for value in values:
        yield (key ,str(value))

  def add(self,key,value):
    """add a new header without deleting old ones"""
    if isinstance(value,list):
      for v in value:
        self.add(key,v)
    elif key in self:
      if isinstance(self[key],list):
        self[key].append(value)
      else:
        self[key] = [self[key],value]
    else :
      self[key] = [value]

      
def abort(code = 500,text = "UnknownError: breaking app"):
  """abort execution and causes a httperror"""
  response.status = code
  response.header['Location'] = url
  raise BreakTheFly("")


def send_file(filename,root,guessmime = True,mimetype = 'text/plain'):

  root = os.path.abspath(root) + '/'
  filename = os.path.normpath(filename).strip('/')
  filename = os.path.join(root,filename)

  if not filename.startswith(root):
    abort(401,"Access denied")
  if not os.path.exists(filename) or not os.path.isfile(filename):
    abort(404,"File not Exists")
  if not os.access(filename,os.R_OK):
    abort(401,"You don't have permission")

  if guessmime:
    guess = mimetypes.guess_type(filename)[0]
    if guess:
      response.content_type = guess
    elif mimetype:
      response.content_type = mimetype

  elif mimetype:
    response.content_type = mimetype

  stats = os.stat(filename)

  if 'Content-Length' not in response.header:
    response.header['Content-Length'] = stats.st_size

  if 'Last-Mofified' not in response.header:
    ts = time.gmtime(stats.st_mtime)
    ts = time.strftime("%a, %d %b %Y %H:%M:%S +0000", ts)
    response.header['Last-Modified'] = ts

  raise BreakTheFly(open(filename), 'r')

def compile_route(route):
  """compile a route string and returns  a precompiled regex obj
  routs contain rgular exp with named groups to support url param
  example : '/user/(?P<id>[0-9]+)' will math '/user/5' with {'id':'5','action':'kiss'}"""

  route = route.strip().lstrip('$^/ ').rstrip('$^ ')

  route = re.sub(r':([a-zA-Z_]+)(?P<uniq>[^\w/])(?P<re>.+?)(?P=uniq)',r'(?P<\1>\g<re>)',route)

  route = re.sub(r':([a-zA-Z_]+)',r'(?P<\1>[^/]+)', route)

  return re.compile('^/%s$' % route)


def match_url(url,method = 'GET'):

  """return the first matching handler and a param dict or (none ,none)"""
  url = '/' + url.strip().lstrip("/")

  route = ROUTES_SIMPLE.get(method,{}).get(url,None)
  if route:
    return (route,{})

  routes = ROUTES_REGEXP.get(method,[])
  for i in  range(len(routes)):
    match = routes[i][0].match(url)
    if match:
      handler = routes[i][1]
      if i > 0 and OPTIMIZER and random.random() <= 0.001:

        #every 1000 request ,swap the mathchg route
        routes[i-1] ,routs[i] = routes[i],routs[i-1]
      return (handler,match.groupdict())
  return (None,None)


def add_route(route,handler,method = 'GET',simple= False):
  """Add a new route
  example:

  def hello():
     return "hello world!
  add_route(r'/hello',hello)"""

  method = method.strip().upper()
  if re.match(r'^/(\w+/)*\w*$',route) or simple:
    ROUTES_SIMPLE.setdefault(method,{})[route] = handler
  else:
    route = compile_route(route)
    ROUTES_REGEXP.setdefault(method,[]).append([route,handler])


def route(url, **kargs):
  """ Decorator for request handler. Same as add_route(url, handler)."""
  def wrapper(handler):
    add_route(url, handler, **kargs)
    return handler
  return wrapper


def validate(**vkargs):
  ''' Validates and manipulates keyword arguments by user defined callables 
  and handles ValueError and missing arguments by raising HTTPError(400) '''

  def decorator(func):
    def wrapper(**kargs):
      for key in vkargs:
        if key not in kargs:
          abort(400, 'Missing parameter: %s' % key)
        try:
          kargs[key] = vkargs[key](kargs[key])
        except ValueError as e:
          abort(400, 'Wrong parameter format for: %s' % key)
      return func(**kargs)
    return wrapper
  return decorator


#error handler

def set_error_handler(code,handler):
  """set a new error handler"""
  code = int(code)
  ERROR_HANDLER[code] = handler

def error(code=500):

  def wrapper(handler):
    set_error_handler(code,handler)
    return handler
  return wrapper


#Server adapter

class ServerAdapter(object):
  def __init__(self,host='127.0.0.1',port=8080,**kargs):
    self.host = host
    self.port = port
    self.options = kargs

  def __repr__(self):
    return "{!s} ({!s} : {:d})".format(self.__class__.__name__,self.host,self.port)

  def run(self,handler):
    pass



class WSGIRefServer(ServerAdapter):
  def run(self, handler):
    from wsgiref.simple_server import make_server
    srv = make_server(self.host, self.port, handler)
    srv.serve_forever()
                                  
  

class CherryPyServer(ServerAdapter):
  def run(self,handler):
    from cherrypy import wsgiserver
    server = wsgiserver.CherryPyWSGIServer((self.host,self.port),handler)
    server.start()

    
class PasteServer(ServerAdapter):

  def run(self,handler):
    from paste import httpserver
    from paste.translogger import TransLogger
    app = TransLogger(Handler)
    httpserver.serve(app,host= self.host,port=str(self.port))
    
  
class FapwsServer(ServerAdapter):
  """extremery fast Webserver using libev """

  def run(self,handler):
    import fapws._evwsgi as evwsgi
    from fapws import base
    import sys
    evwsgi.start(self.host,self.port)
    evwsgi.set_base_module(base)
    def app(environ,start_response):
      environ['wsgi.multiprocess'] = False

      return handler(environ,start_response)
    evwsgi.wsgi_cb(('',app))
    evwsgi.run()

def run(server = CherryPyServer,host= '127.0.0.1', port=8080,optinmize = False,**kargs):
  
  """run fly webserver using builtin wsgiref """
  global OPTIMIZER

  OPTIMIZER = bool(optinmize)
  quiet = bool('quiet' in kargs and kargs['quiet'])

  if isinstance(server,type) and issubclass(server,ServerAdapter):
    server = server(host=host,port=port,**kargs)

  if not isinstance(server,ServerAdapter):
    raise RuntimeError("server must be a subclass of ServerAdapter")

  if not quiet:
    print("Fly server starting (using {!s})...".format(server))
    print('Listening on http://{!s}:{:d}'.format(server.host,server.port))
    print('use Ctrl-C to quit')
    print()

  try:
    server.run(WSGIHandler)
  except KeyboardInterrupt:
    print("shutting down...")

class TemplateError(FlyException): pass
class TemplateNotFoundError(FlyException): pass


class BaseTemplate(object):
  def __init__(self,template='', filename='<template>'):
    self.source = filename
    if self.source != '<template>':
      fp = open(filename)
      template = fp.read()
      fp.close()
    self.parse(template)

  def parse(self,template): raise NotImplementedError
  def render(self,**args): raise NotImplementedError

  @classmethod
  def find(cls, name):
    files = [path % name for path in TEMPLATE_PATH if os.path.isfile(path % name)]
    if files:
      return cls (filename = files[0])
    else:
      raise TemplateError('Template not found {!s}'.format(repr(name)))

class MakoTemplate(BaseTemplate):
  def parse(self,template):
    from mako.template import Template
    self.tpl = Template(template)

  def render(self,**args):
    return self.tpl.render(**args)



class SimpleTemplate(BaseTemplate):

    re_python = re.compile(r'^\s*%\s*(?:(if|elif|else|try|except|finally|for|while|with|def|class)|(include.*)|(end.*)|(.*))')
    re_inline = re.compile(r'\{\{(.*?)\}\}')
    dedent_keywords = ('elif', 'else', 'except', 'finally')

    def parse(self, template):
        indent = 0
        strbuffer = []
        code = []
        self.subtemplates = {}
        class PyStmt(str):
            def __repr__(self): return 'str(' + self + ')'
        def flush():
            if len(strbuffer):
                code.append(" " * indent + "stdout.append(%s)" % repr(''.join(strbuffer)))
                code.append("\n" * len(strbuffer)) # to preserve line numbers 
                del strbuffer[:]
        for line in template.splitlines(True):
            m = self.re_python.match(line)
            if m:
                flush()
                keyword, include, end, statement = m.groups()
                if keyword:
                    if keyword in self.dedent_keywords:
                        indent -= 1
                    code.append(" " * indent + line[m.start(1):])
                    indent += 1
                elif include:
                    tmp = line[m.end(2):].strip().split(None, 1)
                    name = tmp[0]
                    args = tmp[1:] and tmp[1] or ''
                    self.subtemplates[name] = SimpleTemplate.find(name)
                    code.append(" " * indent + "stdout.append(_subtemplates[%s].render(%s))\n" % (repr(name), args))
                elif end:
                    indent -= 1
                    code.append(" " * indent + '#' + line[m.start(3):])
                elif statement:
                    code.append(" " * indent + line[m.start(4):])
            else:
                splits = self.re_inline.split(line) # text, (expr, text)*
                if len(splits) == 1:
                    strbuffer.append(line)
                else:
                    flush()
                    for i in range(1, len(splits), 2):
                        splits[i] = PyStmt(splits[i])
                    code.append(" " * indent + "stdout.extend(%s)\n" % repr(splits))
        flush()
        self.co = compile("".join(code), self.source, 'exec')

    def render(self, **args):
        ''' Returns the rendered template using keyword arguments as local variables. '''
        args['stdout'] = []
        args['_subtemplates'] = self.subtemplates
        eval(self.co, args, globals())
        return ''.join(args['stdout'])


def template(template,template_adapter=SimpleTemplate,**args):
  ''' Returns a string from a template '''
  if template not in TEMPLATES:
    if template.find("\n") == -1 and template.find("{") == -1 and template.find("%") == -1:
      try:
        TEMPLATES[template] = template_adapter.find(template)
      except TemplateNotFoundError : pass


    else:
      TEMPLATES[template] = template_adapter(template)
  if template not in TEMPLATES:
    abort(500,'Template not found')
  args['abort'] = abort
  args['request'] = request
  args['response'] = response

  return TEMPLATES[template].render(**args)


def mako_template(template_name,**args):
  return template(template_name,template_adapter=MakoTemplate,**args)





class FlyBucket(dict):

  def __init__(self,name):
    self.__dict__['name'] = name
    self.__dict__['db'] = dbm.ndbm.open(DB_PATH + '/{!s}.db'.format(name),'c')
    self.__dict__['mmap'] = {}

  def __getattr__(self,key):
    return self.__getitem__(key)

  def __setattr__(self,key,value):
    return self.__setitem__(key,value)

  def __delattr__(self,key):
    return self.__delitem__(key)

  def __getitem__(self,key):
    if key not in self.mmap:
      self.mmap[key] = pickle.loads(self.db[key])
    return self.mmap[key]

  def __setitem__(self,key,value):
    self.mmap[key] = value

  def __delitem__(self,key):
    if key in self.mmap:
      del self.mmap[key]
    del self.db[key]

  def __iter__(self):
    return bool(key in self.mmap or key in self.db)

  def save(self):
    self.close()
    self.__init__(self.name)

  def cloase(self):
    for key in list(self.mmap.keys()):
      pvalue = pickle.dumps(self.mmap[key],pickle.HIGHEST_PROTOCOL)
      if key not in self.db or pvalue != self.db[key]:
        self.db[key] = pvalue

    self.mmap.clear()
    self.db.close()

  def clear(self):
    for key in list(self.db.keys()):
      del self.db[key]

class FlyDB(threading.local):

  def __init__(self):
    self.__dict__['open'] = {}

  def __getattr__(self,key):
    return self.__getitem__(key)

  def __setattr__(self,key,value):
    return self.__setitem__(key,value)

  def __delattr__(self,key):
    return self.__delitem__(key)


  def __getitem__(self,key):
    if key not in self.open:
      self.open[key] = FlyBucket(key)
    return self.open[key]

  def __setitem__(self,key,value):
    if isinstance(value,FlyBucket):
      self.open[key] = value

    elif hasattr(value,'items')  :
      if key not in self.open:
        self.open[key] = FlyBucket(key)
      self.open[key].clear()
      for k,v in list(value.items()):
        self.open[key][k] = v

    else:
      raise AttributeError("only dicts and FlyBucket arr allowed")

  def __delitem__(self,key):
    if key not in self.open:
      self.open[key].clear()
      self.open[key].save()
      del self.open[key]

  def save(self):
    self.close()
    self.__init__(self.name)

  def close(self):
    for db in list(self.open.values()):
      db.close()
    self.open.clear()
    
    
#module initialize

request = Request()
response = Response()

DB_PATH = './'
db = FlyDB()

DEBUG =False
OPTIMIZER = False
TEMPLATE_PATH = ['./{!s}.tpl','./views/{!s}.tpl']
TEMPLATES = {}

ROUTES_SIMPLE = {}
ROUTES_REGEXP = {}
ERROR_HANDLER = {}
HTTP_CODES = {
  100: 'CONTINUE',
  101: 'SWITCHING PROTOCOLS',
  200: 'OK',
  201: 'CREATED',
  202: 'ACCEPTED',
  203: 'NON-AUTHORITIVE INFORMATION',
  204: 'NO CONTENT',
  205: 'RESET CONTENT',
  206: 'PARTIAL CONTENT',
  300: 'MULTIPLE CHOICES',
  301: 'MOVED PERMANENTLY',
  302: 'FOUND',
  303: 'SEE OTHER',
  304: 'NOT MODIFIED',
  305: 'USE PROXY',
  306: 'RESERVED',
  307: 'TEMPORERY REDIRECT',
  400: 'BAD REQUEST',
  401: 'UNAUTHORIZED',
  402: 'PAYMENT REQUIRED',
  403: 'FORBIDDEN',
  404: 'NOT FOUND',
  405: 'METHOD NOT ALLOWED',
  406: 'NOT ACCEPTABLE',
  407: 'PROXY AUTHENTICATION REQUIRED',
  408: 'REQUEST TIMEOUT',
  409: 'CONFLICT',
  410: 'GONE',
  411: 'LENGTH REQUIRED',
  412: 'PRECONDITION FAILED',
  413: 'REQUEST ENTITY TOO LARGE',
  414: 'REQUEST-URI TOO LONG',
  415: 'UNSUPPORTED MEDIA TYPE',
  416: 'REQUESTED RANGE NOT SATISFIABLE',
  417: 'EXPECTATION FAILED',
  500: 'INTERNAL SERVER ERROR',
  501: 'NOT IMPLEMENTED',
  502: 'BAD GATEWAY',
  503: 'SERVICE UNAVAILABLE',
  504: 'GATEWAY TIMEOUT',
  505: 'HTTP VERSION NOT SUPPORTED',
  }


@error(500)
def error500(exception):
  """If an exception is thrown, deal with it and present an error page."""
  if DEBUG:
    return "<br>\n".join(traceback.format_exc(10).splitlines()).replace('  ','&nbsp;&nbsp;')
  else:
    return """<b>Error:</b> Internal server error."""

@error(400)
@error(401)
@error(404)
def error_http(exception):
  status = response.status
  name = HTTP_CODES.get(status,'Unknown').title()
  url = request.path
  """If an exception is thrown, deal with it and present an error page."""
  yield template('<!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML 2.0//EN">'+\
                  '<html><head><title>Error {{status}}: {{msg}}</title>'+\
                  '</head><body><h1>Error {{status}}: {{msg}}</h1>'+\
                  '<p>Sorry, the requested URL {{url}} caused an error.</p>', 
                  status=status,
                  msg=name,
                  url=url
                )
  if hasattr(exception, 'output'):
    yield exception.output
  yield '</body></html>'
  
