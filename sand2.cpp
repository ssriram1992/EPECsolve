#include<iostream>
#include<memory>
#include<vector>

using namespace std;

class my_class
{
	static int i,l;
	int j, k;
	public:
		my_class()
		{
			j=++i;
			k=0;
			cout<<"my_class "<<i<<" constructed\n";
		}
		my_class(my_class &&cop)
		{
			j=++i;k=cop.k;
			cout<<"my_class "<<i<<" constructed as a move of class "<<cop.j<<"\n";
		}
		my_class(const my_class &cop)
		{
			j=++i;k=cop.k;
			cout<<"my_class "<<i<<" constructed as a copy of class "<<cop.j<<"\n";
		}
		~my_class()
		{
			l++;
			cout<<"my_class "<<j<<" destructed with value "<<k<<" and "<<l<<"\n";
		}
		void edit(){++k;};
};

void edit_class(my_class &a)
{
	a.edit();
}

void edit_vec_class(vector<my_class> &vec)
{
	for(auto &a:vec)
		edit_class(a);
}

int my_class::i = 0;
int my_class::l = 0;


int main()
{
	vector<unique_ptr<my_class>> my_vec = vector<unique_ptr<my_class>>(3);
	
	for(auto &elem:my_vec)
		elem = unique_ptr<my_class>(new my_class());

	edit_class(*my_vec.at(1).get());

	return 0;
}
