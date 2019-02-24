function AnimatedText(target,texts,changeInterval,updateInterval,onTextChanged){
	var currentText=parseInt(Math.random()*texts.length);
	var areaText=texts[0];
	this.t1=setInterval(function(){
		var c=parseInt(Math.random()*Math.max(texts[currentText].length,areaText.length));
		var s=texts[currentText][c];
		if(typeof s == 'undefined') s=" ";
		while(areaText.length<c) areaText+=" ";
		var newText=(areaText.slice(0,c)+s+areaText.slice(c+1)).trim();
		var diff=!(newText==areaText);
		areaText=newText;
		if(onTextChanged&&diff) onTextChanged();
		target.innerHTML=areaText.length==0?"&nbsp;":areaText;
	}.bind(this),updateInterval?updateInterval:50);
	this.t2=setInterval(function(){
		currentText=parseInt(Math.random()*texts.length);
	}.bind(this),changeInterval?changeInterval:4000);
}
AnimatedText.prototype={
	constructor:AnimatedText,
	stop:function(){clearInterval(this.t1);clearInterval(this.t2);}
};

$(document).ready(function(){
    text = new AnimatedText(document.getElementById("anitxt"),["Programmer", "Data Science", "Data Analyst", "Student"])
    text.AnimatedText;
})

// $(document).ready(function(){
// 	$(".skills").scroll(function(){
// 		$(".skills").innerHTML = "<div class='container'>"
// 		+"<div class='row'>"
// 		+"<div class='grid-2'>"
// 		+"<h1 style=>Skills</h1>"
// 		+"<ul class='skill-bars'>"
// 		+"<li>"
// 		+"<div class='progress percent90'><span>90%</span></div>"
// 		+"<strong>PYTHON</strong>"    
// 		+"</li>"
// 		+"<li>"
// 		+"<div class='progress percent90'><span>90%</span></div>"
// 		+"<strong>NUMPY</strong>"    
// 		+"</li>"
// 		+"<li><div class='progress percent90'><span>90%</span></div><strong>PANDAS</strong></li>"
// 		+"<li><div class='progress percent85'><span>85%</span></div><strong>MACHINE LEARNING</strong></li>"
// 		+"<li><div class='progress percent70'><span>70%</span></div><strong>R</strong></li>"
// 		+"<li><div class='progress percent75'><span>75%</span></div><strong>SQL</strong></li>"
// 		+"</ul></div></div></div>"
// 	})
// })