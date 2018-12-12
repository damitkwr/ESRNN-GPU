echo "Please enter your email:"
read email

ssh-keygen -t rsa -C $email

cat ~/.ssh/id_rsa.pub

echo "Press enter when you've copied added this to git repo":
read space

git clone git@github.com:damitkwr/11785ESRNN.git

pip install tensorflow

