
# docker : quay.io/pypa/manylinux2010_x86_64
python3.8 setup.py sdist bdist_wheel

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

python3.8 -m twine upload --repository pypi dist/*
