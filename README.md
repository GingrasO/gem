# TRIQS ghostGA

About
-----

This is an implement of the ghost-Gutwiller approximation (ghost-GA) based
on the TRIQS library [1].

Initial Setup
-------------

To install this package, run the following commands in order:

```bash
git clone https://github.com/TRIQS/hartree_fock.git

mkdir ghostGA.build && cd ghostGA.build

cmake ../ghostGA

make
make test
make install
```

### Merging triqs_ghostGA skeleton updates ###

You can merge future changes to the triqs_ghostGA skeleton into your project with the following commands

```bash
git remote update
git merge triqs_ghostGA_remote/python_only -X ours -m "Merge latest triqs_ghostGA skeleton changes"
```

If you should encounter any conflicts resolve them and `git commit`.
Finally we repeat the replace and rename command from the initial setup.

```bash
./share/replace_and_rename.py appname
git commit --amend
```

Now you can compare against the previous commit with: 
```bash
git diff prev_git_hash
````

### Optional ###
----------------

* Add your email address to the bottom section of `Jenkinsfile` for Jenkins CI notification emails
```
End of build log:
\${BUILD_LOG,maxLines=60}
    """,
    to: 'user@domain.org',
```
