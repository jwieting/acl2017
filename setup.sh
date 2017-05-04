mkdir data
cd data

#download simpwiki data
wget http://www.cs.pomona.edu/~dkauchak/simplification/data.v2/sentence-aligned.v2.tar.gz
tar -xvf sentence-aligned.v2.tar.gz
mv sentence-aligned.v2/* .
rm README
rm sentence-aligned.v2.tar.gz
rm -Rf sentence-aligned.v2

#download pre-processed data, ppdb datasets, word embeddings, etc.
wget http://ttic.uchicago.edu/~wieting/acl2017-demo.zip
unzip -j acl2017-demo.zip
rm acl2017-demo.zip