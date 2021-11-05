from utils.util import autocvt, setcfg


def test_autocvt():
    assert 0 == autocvt('0')
    assert -1 == autocvt('-1')
    assert 8 == autocvt('8')
    assert 0.999 == autocvt('0.999')
    assert -0.9 == autocvt('-0.9')
    assert True == autocvt('True')
    assert False == autocvt('False')
    assert '0.9oi90' == autocvt('0.9oi90')
    assert 'TrueFalse' == autocvt('TrueFalse')
    assert '3.True' == autocvt('3.True')
    assert [0, 1, 2] == autocvt('[0, 1, 2 ]')
    assert [True, 1, 'pop'] == autocvt('[True, 1, pop ]')


def test_setcfg():
    assert {'a': 0} == setcfg({'a': 1}, 'a:0')
    assert {'a': {'b': 0.90}} == setcfg({'a': {'b': 1}}, 'a/b:0.90')
    assert {'a': {'b': 0}, 'c': True} == setcfg({'a': {'b': 1}, 'c': False}, 'a/b:0+c:True')
    assert {'a': {'b': 0, 'd': {'e': 'hi'}}, 'c': True} == \
           setcfg({'a': {'b': 1, 'd': {'e': 'hello'}}, 'c': False}, 'a/b:0+c:True+a/d/e:hi')