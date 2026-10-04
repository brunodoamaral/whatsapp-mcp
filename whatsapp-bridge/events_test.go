package main

import (
	"testing"

	"go.mau.fi/whatsmeow/types"
)

func TestDirectChatJID(t *testing.T) {
	list := types.NewJID("1571572178", types.BroadcastServer)
	senderAD := types.JID{User: "77464181153804", Device: 3, Server: types.HiddenUserServer}
	sender := types.NewJID("77464181153804", types.HiddenUserServer)
	me := types.NewJID("5521000000000", types.DefaultUserServer)
	group := types.NewJID("120363027748990179", types.GroupServer)

	cases := []struct {
		name string
		src  types.MessageSource
		want types.JID
	}{
		{"incoming list message goes to sender DM", types.MessageSource{Chat: list, Sender: senderAD}, sender},
		{"own send to own list stays in list", types.MessageSource{Chat: list, Sender: me, IsFromMe: true}, list},
		{"own send with list owner set stays in list", types.MessageSource{Chat: list, Sender: me, IsFromMe: true, BroadcastListOwner: me}, list},
		{"status is not remapped", types.MessageSource{Chat: types.StatusBroadcastJID, Sender: senderAD}, types.StatusBroadcastJID},
		{"DM unchanged", types.MessageSource{Chat: sender, Sender: senderAD}, sender},
		{"group unchanged", types.MessageSource{Chat: group, Sender: senderAD, IsGroup: true}, group},
	}
	for _, c := range cases {
		got := directChatJID(types.MessageInfo{MessageSource: c.src})
		if got != c.want {
			t.Errorf("%s: got %s, want %s", c.name, got, c.want)
		}
	}
}
