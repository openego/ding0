DINGO
=====
DIstribution Network GeneratOr

Required packages
-----------------

* networkx
* geopy
* psycopg2

To install all above listed required package use

```
sudo pip3 install networkx geopy
```

The package `psycopg2` maybe requires postgresql-server-dev-<verion>.<number> to
be installed. If so, first install this via package management system. If you
have Ubuntu as OS just issue

```
sudo apt-g  install postgresql-server-dev-<verion>.<number>
```

and afterwards

```
sudo pip3 install psycopg2
```