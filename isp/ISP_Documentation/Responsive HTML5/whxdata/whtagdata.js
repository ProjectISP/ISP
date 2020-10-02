(function() {
var tagCombinations =  [] ,
  tags =  [],
  caption = "",
  type = "radio",
  defFilter = {};

window.rh.model.publish("p.tag_combinations", tagCombinations, { sync:true });
window.rh.model.publish("temp.data", {"tags": tags, "caption": caption, "type": type , "default": defFilter}, { sync:true });
})();
