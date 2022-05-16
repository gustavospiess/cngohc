#!/usr/bin/env python

'''Tests for `hogc` package.'''

import pytest

from hogc import hogc


@pytest.fixture
def response():
    '''Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    '''
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    '''Sample pytest test function with the pytest fixture as an argument.'''
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_has_doc():
    '''It's ridiculous, but it's what I have'''
    assert hogc.__doc__


def test_has_main():
    '''It's ridiculous, but it's what I have'''
    assert hogc.main
    assert not hogc.main()
