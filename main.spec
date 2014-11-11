# -*- mode: python -*-
a = Analysis(['main.py'],
             pathex=['/home/flipajs/Dropbox/PycharmProjects/ants'],
             hiddenimports=[],
             hookspath=None,
             runtime_hooks=None)
pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='main',
          debug=False,
          strip=None,
          upx=True,
          console=True )
