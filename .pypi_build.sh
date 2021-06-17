
# docker : quay.io/pypa/manylinux2010_x86_64
for PY in "python3.6" "python3.7" "python3.8" ; do
	$PY setup.py sdist bdist_wheel
done

cd dist
for k in `ls | grep whl$` ;  do
        echo $k
        auditwheel repair $k
        rm $k
done
cd wheelhouse
mv ./*.whl ../
rm -rf wheelhouse

cd ..

python3.6 -m twine upload --repository pypi dist/*
