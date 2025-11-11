安装指南
========

环境要求
--------

* Python 3.8+
* networkx
* pip

使用 pip 安装
-------------

.. code-block:: bash

   pip install pynetim

从源码安装
----------

.. code-block:: bash

   git clone https://github.com/zzzkhj/PyNetIM.git
   cd PyNetIM
   pip install -e .

验证安装
--------

.. code-block:: python

   import pynetim
   print(pynetim.__version__)