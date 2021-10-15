#include <bits/stdc++.h>
using namespace std;

//Geo Info
struct Region{
	int min_x,max_x,min_y,max_y,area;
	void Output(){
		cout<<min_x<<" "<<max_x<<" "<<min_y<<" "<<max_y<<" "<<area<<endl;
	}
}Geo[20];

//Tags Info
struct Tags{
	string name;
	int num,sz;
	int idx[10];
	void Output(){
		cout<<num<<" "<<name<<" ";
		for(int i=1;i<=sz;i++){
			cout<<idx[i]<<" ";
		}
		cout<<endl;
	}
}tags[100];

int TagsNum = 60;
int DataRange = 10000; 

int main()
{
	string city,TagName;
	int min_x,max_x,min_y,max_y,area;
	ifstream in;
	in.open("SDCities.txt");
	//Input Geo Info
	for(int i=1;i<=17;i++){
		in>>city;
		in>>Geo[i].min_x>>Geo[i].max_x>>Geo[i].min_y>>Geo[i].max_y>>Geo[i].area;
	}
	in.close();
	/*
	for(int i=1;i<=17;i++){
		Geo[i].Output();
	}
	*/
	freopen("TagsInfo2.txt","r",stdin);
	freopen("CityTags2.txt","w",stdout);
	// Total Points
	int Count=0;
	//Input Tags Region 
	vector<int> a;
	string str;
	stringstream ss;
	bool flag=false;
	for(int i=1;i<=TagsNum;i++){
		str="";
		getline(cin,str);
		ss.clear();
		ss<<str;
		ss>>tags[i].num;
		ss>>tags[i].name;
		while(ss){
			int t;
			ss>>t;
			a.push_back(t);
		}
		a.pop_back();
		tags[i].sz=a.size();
		for(int j=0;j<a.size();j++){
			tags[i].idx[j+1]=a[j];
		}
		a.clear(); 
		Count+=tags[i].num*tags[i].sz;
		//tags[i].Output();
	}
	//Generate Data
	for(int i=1;i<=TagsNum;i++){
		for(int j=1;j<=tags[i].sz;j++){
			// Frequency
			int num=((double)tags[i].num/(double)Count)*DataRange;
			//cout<<num<<endl; 
			random_device rd;
			mt19937_64 eng(rd());
			// Coordinate
			uniform_int_distribution<unsigned long long> distrx(Geo[tags[i].idx[j]].min_x, Geo[tags[i].idx[j]].max_x); 
			uniform_int_distribution<unsigned long long> distry(Geo[tags[i].idx[j]].min_y, Geo[tags[i].idx[j]].max_y); 
			// Tags x y frequency
			for(int k=0;k<num;k++){
				cout<<tags[i].name<<" "<<distrx(eng)<<" "<<distry(eng)<<" "<<tags[i].idx[j]<<endl;
			}
		}
	}
	return 0;
}
